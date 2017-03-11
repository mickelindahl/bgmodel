/*
 *  poisson_periodically.cpp
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

#include "poisson_periodically.h"
#include "network.h"
#include "dict.h"
#include "doubledatum.h"
#include "dictutils.h"
#include "exceptions.h"

using namespace nest;

/* ----------------------------------------------------------------
 * Default constructors defining default parameter
 * ---------------------------------------------------------------- */

mynest::poisson_periodically::Parameters_::Parameters_()
  : period_low_(500.0),
    period_high_(500.0),
	rate_low_(10.0),
    rate_high_(20.0) // pA
{}


/* ----------------------------------------------------------------
 * Parameter extraction and manipulation functions
 * ---------------------------------------------------------------- */

void mynest::poisson_periodically::Parameters_::get(DictionaryDatum &d) const
{
  def<nest::double_t>(d, "period_low", period_low_);
  def<nest::double_t>(d, "period_high", period_high_);
  def<nest::double_t>(d, "rate_low", rate_low_);
  def<nest::double_t>(d, "rate_high", rate_high_);

}

void mynest::poisson_periodically::Parameters_::set(const DictionaryDatum& d)
{
	updateValue<nest::double_t>(d, "period_low", period_low_);
	updateValue<nest::double_t>(d, "period_high", period_high_);
	updateValue<nest::double_t>(d, "rate_low", rate_low_);
	updateValue<nest::double_t>(d, "rate_high", rate_high_);

  if (( rate_low_ < 0 ) || ( rate_high_ < 0 ))
    throw nest::BadProperty("The rate cannot be negative.");
}


/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

mynest::poisson_periodically::poisson_periodically()
  : Node(),
    device_(),
    P_()
{}

mynest::poisson_periodically::poisson_periodically(const mynest::poisson_periodically& n)
  : Node(n),
    device_(n.device_),
    P_(n.P_)
{}


/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void mynest::poisson_periodically::init_state_(const nest::Node& proto)
{
  const mynest::poisson_periodically& pr = downcast<mynest::poisson_periodically>(proto);

//  device_.init_state(pr.device_);
}

void mynest::poisson_periodically::init_buffers_()
{
  device_.init_buffers();
}

void mynest::poisson_periodically::calibrate()
{
  device_.calibrate();

  // rate_ is in Hz, dt in ms, so we have to convert from s to ms
  V_.poisson_dev_.set_lambda(nest::Time::get_resolution().get_ms() * P_.rate_low_ * 1e-3);
  V_.period_= P_.rate_low_+P_.rate_high_;
}


/* ----------------------------------------------------------------
 * Update function and event hook
 * ---------------------------------------------------------------- */

void mynest::poisson_periodically::update(nest::Time const & T,
		const nest::long_t from, const nest::long_t to)
{
  assert(to >= 0 && (nest::delay) from < nest::Scheduler::get_min_delay());
  assert(from < to);



  if ( P_.rate_low_ <= 0 )
    return;

  if ( P_.rate_high_ <= 0 )
    return;

  for ( nest::long_t lag = from ; lag < to ; ++lag )
  {
	  // Change rate periodically
	  if ((lag % V_.period_) == P_.period_low_)
	  {
//		V_.poisson_dev_.set_lambda(P_.rate_high_);
	  	cout << 300 << endl;
	  	cout << P_.rate_high_ << endl;
	  }

	  if ((lag % V_.period_) == 0)
	  {
//		V_.poisson_dev_.set_lambda(P_.rate_low_);
	  	cout << 400 << endl;
	  	cout << P_.rate_low_ << endl;

	  }
	  cout << 200 << endl;
	  cout << lag  << endl;
	  cout << lag % V_.period_ << endl;
//	if ( !device_.is_active( T + nest::Time::step(lag) ) )
//      continue;  // no spike at this lag

    nest::DSSpikeEvent se;
    network()->send(*this, se, lag);
  }
}

void mynest::poisson_periodically::event_hook(nest::DSSpikeEvent& e)
{
  librandom::RngPtr rng = net_->get_rng(get_thread());
  nest::ulong_t n_spikes = V_.poisson_dev_.uldev(rng);

  if ( n_spikes > 0 ) // we must not send events with multiplicity 0
  {
    e.set_multiplicity(n_spikes);
    e.get_receiver().handle(e);
  }
}


