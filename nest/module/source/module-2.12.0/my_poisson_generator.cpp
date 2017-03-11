/*
 *  my_poisson_generator.cpp
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

#include "my_poisson_generator.h"

// Includes from nestnest::kernel:
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


mynest::my_poisson_generator::Parameters_::Parameters_()
  : rate_(0.0    )  // pA
{}


/* ----------------------------------------------------------------
 * Parameter extraction and manipulation functions
 * ---------------------------------------------------------------- */

void mynest::my_poisson_generator::Parameters_::get(DictionaryDatum &d) const
{
  def<double>(d, nest::names::rate, rate_);
}

void mynest::my_poisson_generator::Parameters_::set(const DictionaryDatum& d)
{
  updateValue<double>(d, nest::names::rate, rate_);
  if ( rate_ < 0 )
    throw nest::BadProperty("The rate cannot be negative.");
}


/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

mynest::my_poisson_generator::my_poisson_generator()
  : Node(),
    device_(),
    P_()
{}

mynest::my_poisson_generator::my_poisson_generator(const my_poisson_generator& n)
  : Node(n),
    device_(n.device_),
    P_(n.P_)
{}


/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void mynest::my_poisson_generator::init_state_(const Node& proto)
{
  const my_poisson_generator& pr = downcast<my_poisson_generator>(proto);

  device_.init_state(pr.device_);
}

void mynest::my_poisson_generator::init_buffers_()
{
  device_.init_buffers();
}

void mynest::my_poisson_generator::calibrate()
{
  device_.calibrate();

  // rate_ is in Hz, dt in ms, so we have to convert from s to ms
  V_.poisson_dev_.set_lambda(nest::Time::get_resolution().get_ms() * P_.rate_ * 1e-3);
}


/* ----------------------------------------------------------------
 * Update function and event hook
 * ---------------------------------------------------------------- */

void mynest::my_poisson_generator::update(nest::Time const & T,
		const long from, const long to)
{
  assert(
  to >= 0 && (nest::delay) from < nest::kernel().connection_manager.get_min_delay());
  assert(from < to);

  if ( P_.rate_ <= 0 )
    return;

  for ( long lag = from ; lag < to ; ++lag )
  {
    if ( !device_.is_active( T + nest::Time::step(lag) ) )
      continue;  // no spike at this lag

    nest::DSSpikeEvent se;
//    nest::SpikeEvent se;

    nest::kernel().event_delivery_manager.send( *this, se, lag );
    //network()->send(*this, se, lag);
  }
}

void
mynest::my_poisson_generator::event_hook(nest::DSSpikeEvent& e)
//void mynest::my_poisson_generator::event_hook(nest::SpikeEvent& e)
{
  librandom::RngPtr rng = nest::kernel().rng_manager.get_rng(get_thread());
  long n_spikes = V_.poisson_dev_.ldev(rng);

  if ( n_spikes > 0 ) // we must not send events with multiplicity 0
  {
    e.set_multiplicity(n_spikes);
    e.get_receiver().handle(e);
  }
}
