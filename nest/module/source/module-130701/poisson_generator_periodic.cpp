/*
 *  poisson_generator_periodic.cpp
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

#include "poisson_generator_periodic.h"
#include "network.h"
#include "dict.h"
#include "doubledatum.h"
#include "dictutils.h"
#include "exceptions.h"

#include <functional>   // std::modulus

/* ----------------------------------------------------------------
 * Default constructors defining default parameter
 * ---------------------------------------------------------------- */

using namespace nest;

mynest::poisson_generator_periodic::Parameters_::Parameters_()
  : period_first_(500.0),
    period_second_(500.0),
	rate_first_(10.0),
    rate_second_(20.0) // pA
{}


/* ----------------------------------------------------------------
 * Parameter extraction and manipulation functions
 * ---------------------------------------------------------------- */

void mynest::poisson_generator_periodic::Parameters_::get(DictionaryDatum &d) const
{
  def<nest::double_t>(d, "period_first", period_first_);
  def<nest::double_t>(d, "period_second", period_second_);
  def<nest::double_t>(d, "rate_first", rate_first_);
  def<nest::double_t>(d, "rate_second", rate_second_);

}

void mynest::poisson_generator_periodic::Parameters_::set(const DictionaryDatum& d)
{
	updateValue<nest::double_t>(d, "period_first", period_first_);
	updateValue<nest::double_t>(d, "period_second", period_second_);
	updateValue<nest::double_t>(d, "rate_first", rate_first_);
	updateValue<nest::double_t>(d, "rate_second", rate_second_);

  if (( rate_first_ < 0 ) || ( rate_second_ < 0 ))
    throw nest::BadProperty("The rate cannot be negative.");
}


/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

mynest::poisson_generator_periodic::poisson_generator_periodic()
  : Node(),
    device_(),
    P_()
{}

mynest::poisson_generator_periodic::poisson_generator_periodic(const mynest::poisson_generator_periodic& n)
  : Node(n),
    device_(n.device_),
    P_(n.P_)
{}


/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void mynest::poisson_generator_periodic::init_state_(const Node& proto)
{
  const mynest::poisson_generator_periodic& pr = downcast<mynest::poisson_generator_periodic>(proto);

  device_.init_state(pr.device_);
}

void mynest::poisson_generator_periodic::init_buffers_()
{
  device_.init_buffers();
}

void mynest::poisson_generator_periodic::calibrate()
{
  device_.calibrate();

  // rate_ is in Hz, dt in ms, so we have to convert from s to ms
  V_.poisson_dev_.set_lambda(nest::Time::get_resolution().get_ms() * P_.rate_first_ * 1e-3);
  V_.period_= P_.period_first_+P_.period_second_;
}


/* ----------------------------------------------------------------
 * Update function and event hook
 * ---------------------------------------------------------------- */

void mynest::poisson_generator_periodic::update(nest::Time const & T,
		const nest::long_t from, const nest::long_t to)
{
  assert(to >= 0 && (nest::delay) from < nest::Scheduler::get_min_delay());
  assert(from < to);



  if (( P_.rate_first_ <= 0 ) && ( P_.rate_second_ <= 0 ))
    return;



  // Change rate periodically
  nest::long_t period=(nest::long_t) T.get_ms() % (nest::long_t) V_.period_;

  if (period == (nest::long_t) P_.period_first_)
  {
	V_.poisson_dev_.set_lambda(nest::Time::get_resolution().get_ms() *P_.rate_second_* 1e-3);
  }

  if (period == 0)
  {
	V_.poisson_dev_.set_lambda(nest::Time::get_resolution().get_ms() *P_.rate_first_* 1e-3);
  }

//  cout << (nest::long_t) V_.period_ % (nest::long_t) T.get_ms()  << endl;


  for ( nest::long_t lag = from ; lag < to ; ++lag )
  {
	if ( !device_.is_active( T + nest::Time::step(lag) ) )
      continue;  // no spike at this lag

    nest::DSSpikeEvent se;
    network()->send(*this, se, lag);
  }
}

void mynest::poisson_generator_periodic::event_hook(nest::DSSpikeEvent& e)
{
  librandom::RngPtr rng = net_->get_rng(get_thread());
  nest::ulong_t n_spikes = V_.poisson_dev_.uldev(rng);

  if ( n_spikes > 0 ) // we must not send events with multiplicity 0
  {
    e.set_multiplicity(n_spikes);
    e.get_receiver().handle(e);
  }
}


