/*
 *  poisson_generator_dynamic.cpp
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

/* for Debugging */
#include <iostream>
using namespace std;
#include <functional>   // std::modulus

#include "poisson_generator_dynamic.h"

// Includes from nestkernel:
#include "event_delivery_manager_impl.h"
#include "exceptions.h"
#include "kernel_manager.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"

//#include "network.h" //gone


/* ----------------------------------------------------------------
 * Default constructors defining default parameter
 * ---------------------------------------------------------------- */

using namespace nest;

mynest::poisson_generator_dynamic::Parameters_::Parameters_()
  : timings_(),
    rates_() // pA
{}

mynest::poisson_generator_dynamic::Parameters_::Parameters_(const Parameters_& op)
  :timings_(op.timings_),
   rates_(op.rates_) // pA
{}

mynest::poisson_generator_dynamic::State_::State_()
  : position_(0)
{}

/* ----------------------------------------------------------------
 * Parameter extraction and manipulation functions
 * ---------------------------------------------------------------- */

void mynest::poisson_generator_dynamic::Parameters_::get(DictionaryDatum &d) const
{
//	  const size_t n_timings = timings_.size();
//	  const size_t n_rates_ = rates_.size();
//	  assert(   ( timings_ && n_timings == n_rates_ )
//		 || (!timings_ && n_rates_ == 0        ) );


	  (*d)["timings"] = timings_;
	  (*d)["rates"] = rates_;
}

void mynest::poisson_generator_dynamic::Parameters_::set(const DictionaryDatum& d)
{
	const bool updated_timings = d->known("timings");
	const bool updated_rates = d->known("rates");
	if (updated_timings)
	{
		std::vector<double> timings = getValue<std::vector<double> >(d->lookup("timings"));
		timings_.swap(timings);
	}
	if (updated_rates)
	{
		std::vector<double> rates = getValue<std::vector<double> >(d->lookup("rates"));
		rates_.swap(rates);
	}
}



/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

mynest::poisson_generator_dynamic::poisson_generator_dynamic()
  : Node(),
    device_(),
    P_()
{}

mynest::poisson_generator_dynamic::poisson_generator_dynamic(const mynest::poisson_generator_dynamic& n)
  : Node(n),
    device_(n.device_),
    P_(n.P_)
{}


/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void mynest::poisson_generator_dynamic::init_state_(const Node& proto)
{
  const mynest::poisson_generator_dynamic& pr = downcast<mynest::poisson_generator_dynamic>(proto);

  device_.init_state(pr.device_);
}

void mynest::poisson_generator_dynamic::init_buffers_()
{
  device_.init_buffers();
}

void mynest::poisson_generator_dynamic::calibrate()
{
  device_.calibrate();

//  cout << "Calibrate"<<endl;
//  cout << S_.position_ << endl;
//  cout << P_.rates_.size() << endl;
  // rate_ is in Hz, dt in ms, so we have to convert from s to ms
  if (P_.rates_.size())
	V_.poisson_dev_.set_lambda(nest::Time::get_resolution().get_ms()
			* P_.rates_[S_.position_] * 1e-3);
}


/* ----------------------------------------------------------------
 * Update function and event hook
 * ---------------------------------------------------------------- */

void mynest::poisson_generator_dynamic::update(nest::Time const & T,
		const long from, const long to)
{
  assert(to >= 0 && (nest::delay) from < nest::kernel().connection_manager.get_min_delay());
  assert(from < to);


  if (!P_.rates_.size())
	  return;

  if (P_.timings_.size()-1>S_.position_)
	  if (P_.timings_[S_.position_+1]<= T.get_ms())
	  {
		  S_.position_++;

  	  	  if (P_.rates_[S_.position_]<=0)
  	  		  return;

  	  	  V_.poisson_dev_.set_lambda(nest::Time::get_resolution().get_ms() * P_.rates_[S_.position_]* 1e-3);
	  }

  if (P_.rates_[S_.position_] <= 0)
	  return;

  for ( long lag = from ; lag < to ; ++lag )
  {
	if ( !device_.is_active( T + nest::Time::step(lag) ) )
      continue;  // no spike at this lag

    nest::DSSpikeEvent se;
    nest::kernel().event_delivery_manager.send(*this, se, lag);
  }
}

void mynest::poisson_generator_dynamic::event_hook(nest::DSSpikeEvent& e)
{
  librandom::RngPtr rng = nest::kernel().rng_manager.get_rng(get_thread());
  long n_spikes = V_.poisson_dev_.ldev(rng);

  if ( n_spikes > 0 ) // we must not send events with multiplicity 0
  {
    e.set_multiplicity(n_spikes);
    e.get_receiver().handle(e);
  }
}


