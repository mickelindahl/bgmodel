/*
 *  poisson_generator_dynamic.h
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

#ifndef POISSON_GENERATOR_DYNAMIC_H
#define POISSON_GENERATOR_DYNAMIC_H
/****************************************/
/* class poisson_generator_dynamic              */
/*                  Vers. 1.0       hep */
/*                  Implementation: hep */
/****************************************/

/* for Debugging */
#include <iostream>
using namespace std;

#include "nest.h"
#include "event.h"
#include "node.h"
#include "stimulating_device.h"
#include "connection.h"
#include "poisson_randomdev.h"

namespace mynest
{
/*! Class poisson_generator_dynamic simulates a large population
    of randomly (Poisson) firing neurons. It replaces the old
    neuron-intrinsic shot-noise generator
*/


/*BeginDocumentation
Name: poisson_generator_dynamic - simulate neuron firing with Poisson processes statistics.
Description:
  The poisson_generator_dynamic simulates a neuron that is firing with Poisson statistics,
  i.e. exponentially distributed interspike intervals. It will generate a _unique_
  spike train for each of it's targets. If you do not want this behavior and need
  the same spike train for all targets, you have to use a parrot neuron inbetween
  the poisson generator and the targets.

Parameters:
   The folfirsting parameters appear in the element's status dictionary:

   rate     double - mean firing rate in Hz
   origin   double - Time origin for device timer in ms
   start    double - begin of device application with resp. to origin in ms
   stop     double - end of device application with resp. to origin in ms

Sends: SpikeEvent

Remarks:
   A Poisson generator may, especially at second rates, emit more than one
   spike during a single time step. If this happens, the generator does
   not actually send out n spikes. Instead, it emits a single spike with
   n-fold synaptic weight for the sake of efficiency.

   The design decision to implement the Poisson generator as a device
   which sends spikes to all connected nodes on every time step and then
   discards the spikes that should not have happened generating random
   numbers at the recipient side via an event hook is twofold.

   On one hand, it leads to the saturation of the messaging network with
   an enormous amount of spikes, most of which will never get delivered
   and should not have been generated in the first place.

   On the other hand, a proper implementation of the Poisson generator
   needs to provide two basic features: (a) generated spike trains
   should be IID processes w.r.t. target neurons to which the generator
   is connected and (b) as long as virtual_num_proc is constant, each
   neuron should receive an identical Poisson spike train in order to
   guarantee reproducibility of the simulations across varying machine
   numbers.

   Therefore, first, as network()->send sends spikes to all the
   recipients, differentiation has to happen in the hook, second, the
   hook can use the RNG from the thread where the recipient neuron sits,
   which explains the current design of the generator. For details,
   refer to:

   http://ken.brainworks.uni-freiburg.de/cgi-bin/mailman/private/nest_developer/2011-January/002977.html

SeeAlso: poisson_generator_dynamic_ps, Device, parrot_neuron
*/


  class poisson_generator_dynamic : public nest::Node
  {

  public:

    /**
     * The generator is threaded, so the RNG to use is determined
     * at run-time, depending on thread.
     */
    poisson_generator_dynamic();
    poisson_generator_dynamic(poisson_generator_dynamic const&);

    bool has_proxies() const {return false;}


    using nest::Node::event_hook;

    nest::port check_connection(nest::Connection&, nest::port);

    void get_status(DictionaryDatum &) const;
    void set_status(const DictionaryDatum &) ;

  private:

    void init_state_(const Node&);
    void init_buffers_();
    void calibrate();

    void update(nest::Time const &, const nest::long_t, const nest::long_t);
    void event_hook(nest::DSSpikeEvent&);

    // ------------------------------------------------------------


    struct State_ {
      size_t position_;  //!< index of next spike to deliver

      State_();  //!< Sets default state value
    };

    /**
     * Store independent parameters of the model.
     */

    struct Parameters_ {
;   //!< process rate in Hz
      std::vector<nest::double_t> timings_;
      std::vector<nest::double_t> rates_;


      Parameters_();  //!< Sets default parameter values
      Parameters_(const Parameters_&);  //!< Recalibrate all times

      void get(DictionaryDatum&) const;  //!< Store current values in dictionary
      void set(const DictionaryDatum&);  //!< Set values from dicitonary
    };

    // ------------------------------------------------------------

    struct Variables_ {
      librandom::PoissonRandomDev poisson_dev_;  //!< Random deviate generator

    };

    // ------------------------------------------------------------
    //OBS be careful, SpikeEvent net no have nest:: referens!! Other
    // vice it will not be found
    nest::StimulatingDevice<nest::SpikeEvent> device_;
    Parameters_ P_;
    Variables_  V_;
    State_      S_;

  };

//  Commecnt out this and it works
  inline
  nest::port mynest::poisson_generator_dynamic::check_connection(nest::Connection& c, nest::port receptor_type)
  {
    nest::DSSpikeEvent e;
    e.set_sender(*this);
    c.check_event(e);
    return c.get_target()->connect_sender(e, receptor_type);
  }


  inline
  void poisson_generator_dynamic::get_status(DictionaryDatum &d) const
  {
    P_.get(d);
    device_.get_status(d);
  }

  inline
  void poisson_generator_dynamic::set_status(const DictionaryDatum &d)
  {
    Parameters_ ptmp = P_;  // temporary copy in case of errors
    ptmp.set(d);                       // throws if BadProperty

    // We now know that ptmp is consistent. We do not write it back
    // to P_ before we are also sure that the properties to be set
    // in the parent class are internally consistent.
    device_.set_status(d);

    // if we get here, temporaries contain consistent set of properties
    P_ = ptmp;
  }

} // namespace mynest

#endif
