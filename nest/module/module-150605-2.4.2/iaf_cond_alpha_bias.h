/*
 *  iaf_cond_alpha_bias.h
 *
 *  This file is part of NEST
 *
 *  Copyright (C) 2005-2009 by
 *  The NEST Initiative
 *
 *  See the file AUTHORS for details.
 *
 *  Permission is granted to compile and modify
 *  this file for non-commercial use.
 *  See the file LICENSE for details.
 *
 *  written by Philip Tully
 *  first version February 2012
 *
 */

#ifndef IAF_COND_ALPHA_BIAS_H
#define IAF_COND_ALPHA_BIAS_H

#include "config.h"

#ifdef HAVE_GSL

#include "nest.h"
#include "event.h"
#include "archiving_node.h"
#include "ring_buffer.h"
#include "connection.h"
#include "universal_data_logger.h"
#include "recordables_map.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

/* BeginDocumentation
Name: iaf_cond_alpha_bias - Simple conductance based leaky integrate-and-fire neuron model.
                            Incorporates Bayesian Bias dynamics depending on incoming spike events.

Description:
iaf_cond_alpha is an implementation of a spiking neuron using IAF dynamics with
conductance-based synapses. Incoming spike events induce a post-synaptic change 
of conductance modelled by an alpha function. The alpha function 
is normalised such that an event of weight 1.0 results in a peak current of 1 nS
at t = tau_syn.

Parameters: 
The following parameters can be set in the status dictionary.

V_m        double - Membrane potential in mV 
E_L        double - Leak reversal potential in mV.
C_m        double - Capacity of the membrane in pF
t_ref      double - Duration of refractory period in ms. 
V_th       double - Spike threshold in mV.
V_reset    double - Reset potential of the membrane in mV.
E_ex       double - Excitatory reversal potential in mV.
E_in       double - Inhibitory reversal potential in mV.
g_L        double - Leak conductance in nS;
tau_syn_ex double - Rise time of the excitatory synaptic alpha function in ms.
tau_syn_in double - Rise time of the inhibitory synaptic alpha function in ms.
I_e        double - Constant input current in pA.
tau_j--\
tau_e   >- double - Postsynaptic trace time constants
tau_p--/
kappa      double - 'print now' signal

Sends: SpikeEvent

Receives: SpikeEvent, CurrentEvent, DataLoggingRequest

References: 

Author: Tully, Philip

SeeAlso: iaf_cond_alpha, iaf_cond_exp, iaf_cond_alpha_mc
*/

namespace mynest
{
  /**
   * Function computing right-hand side of ODE for GSL solver.
   * @note Must be declared here so we can befriend it in class.
   * @note Must have C-linkage for passing to GSL. Internally, it is
   *       a first-class C++ function, but cannot be a member function
   *       because of the C-linkage.
   * @note No point in declaring it inline, since it is called
   *       through a function pointer.
   * @param void* Pointer to model neuron instance.
   */
  extern "C"
  int iaf_cond_alpha_bias_dynamics (double, const double*, double*, void*);

  /**
   * Integrate-and-fire neuron model with two conductance-based synapses.
   *
   * @note Per 2009-04-17, this class has been revised to our newest 
   *       insights into class design. Please use THIS CLASS as a reference
   *       when designing your own models with nonlinear dynamics.
   *       One weakness of this class is that it distinguishes between
   *       inputs to the two synapses by the sign of the synaptic weight.
   *       It would be better to use receptor_types, cf iaf_cond_alpha_mc.
   */
  class iaf_cond_alpha_bias : public nest::Archiving_Node
  {
    
    // Boilerplate function declarations --------------------------------

  public:
    
    iaf_cond_alpha_bias();
    iaf_cond_alpha_bias(const iaf_cond_alpha_bias&);
    ~iaf_cond_alpha_bias();

    /*
     * Import all overloaded virtual functions that we
     * override in this class.  For background information, 
     * see http://www.gotw.ca/gotw/005.htm.
     */

    using nest::Node::connect_sender;
    using nest::Node::handle;

    nest::port check_connection(nest::Connection&, nest::port);

    nest::port connect_sender(nest::SpikeEvent &, nest::port);
    nest::port connect_sender(nest::CurrentEvent &, nest::port);
    nest::port connect_sender(nest::DataLoggingRequest &, nest::port);
    
    void handle(nest::SpikeEvent &);
    void handle(nest::CurrentEvent &);
    void handle(nest::DataLoggingRequest &); 
        
    void get_status(DictionaryDatum &) const;
    void set_status(const DictionaryDatum &);

  private:
    void init_node_(const Node& proto);
    void init_state_(const Node& proto);
    void init_buffers_();
    void calibrate();
    void update(nest::Time const &, const nest::long_t, const nest::long_t);

    // END Boilerplate function declarations ----------------------------

    // Friends --------------------------------------------------------

    // make dynamics function quasi-member mynest? or nest? also above
    friend int mynest::iaf_cond_alpha_bias_dynamics(double, const double*, double*, void*);

    // The next two classes need to be friends to access the State_ class/member
    friend class nest::RecordablesMap<iaf_cond_alpha_bias>;
    friend class nest::UniversalDataLogger<iaf_cond_alpha_bias>;

  private:

    // Parameters class ------------------------------------------------- 

    //! Model parameters
    struct Parameters_ {
      nest::double_t V_th;        //!< Threshold Potential in mV
      nest::double_t V_reset;     //!< Reset Potential in mV
      nest::double_t t_ref;       //!< Refractory period in ms
      nest::double_t g_L;         //!< Leak Conductance in nS
      nest::double_t C_m;         //!< Membrane Capacitance in pF
      nest::double_t E_ex;        //!< Excitatory reversal Potential in mV
      nest::double_t E_in;        //!< Inhibitory reversal Potential in mV
      nest::double_t E_L;         //!< Leak reversal Potential (aka resting potential) in mV
      nest::double_t tau_synE;    //!< Synaptic Time Constant Excitatory Synapse in ms
      nest::double_t tau_synI;    //!< Synaptic Time Constant for Inhibitory Synapse in ms
      nest::double_t I_e;         //!< Constant Current in pA
      nest::double_t tau_j;
      nest::double_t tau_e;
      nest::double_t tau_p;
      nest::double_t bias;
      nest::double_t fmax;
      nest::double_t gain;
      nest::double_t epsilon;
      nest::double_t kappa;
  
      Parameters_();        //!< Set default parameter values

      void get(DictionaryDatum&) const;  //!< Store current values in dictionary
      void set(const DictionaryDatum&);  //!< Set values from dicitonary
    };
    
    // State variables class -------------------------------------------- 

    /**
     * State variables of the model.
     * 
     * State variables consist of the state vector for the subthreshold
     * dynamics and the refractory count. The state vector must be a
     * C-style array to be compatible with GSL ODE solvers.
     *
     * @note Copy constructor and assignment operator are required because
     *       of the C-style array.
     */
  public:
    struct State_ {
      
      //! Symbolic indices to the elements of the state vector y
      enum StateVecElems { V_M = 0,           
			   DG_EXC, //1
                           G_EXC,  //2    
			   DG_INH, //3
                           G_INH,  //4
                           Z_J,    //5
                           E_J,    //6
                           P_J,    //7
                           //BIAS,   //8
			   STATE_VEC_SIZE };

      //! state vector, must be C-array for GSL solver
      nest::double_t y[STATE_VEC_SIZE];
  
      //!< number of refractory steps remaining
      nest::int_t    r; 

      //!< Bias calculated from the P_J trace 
      nest::double_t bias;
      nest::double_t epsilon;
      nest::double_t kappa;

      State_(const Parameters_&);  //!< Default initialization
      State_(const State_&);
      State_& operator=(const State_&);

      void get(DictionaryDatum&) const;  //!< Store current values in dictionary

      /**
       * Set state from values in dictionary.
       * Requires Parameters_ as argument to, eg, check bounds.'
       */
      void set(const DictionaryDatum&, const Parameters_&);
    };    
  private:

    // Buffers class -------------------------------------------------------- 

    /**
     * Buffers of the model.
     * Buffers are on par with state variables in terms of persistence,
     * i.e., initalized only upon first Simulate call after ResetKernel
     * or ResetNetwork, but are implementation details hidden from the user.
     */
    struct Buffers_ {
      Buffers_(iaf_cond_alpha_bias&); //!<Sets buffer pointers to 0
      Buffers_(const Buffers_&, iaf_cond_alpha_bias&); //!<Sets buffer pointers to 0

      //! Logger for all analog data
      nest::UniversalDataLogger<iaf_cond_alpha_bias> logger_;

      /** buffers and sums up incoming spikes/currents */
      nest::RingBuffer spike_exc_;
      nest::RingBuffer spike_inh_;
      nest::RingBuffer currents_;

      /* GSL ODE stuff */
      gsl_odeiv_step*    s_;    //!< stepping function
      gsl_odeiv_control* c_;    //!< adaptive stepsize control function
      gsl_odeiv_evolve*  e_;    //!< evolution function
      gsl_odeiv_system   sys_;  //!< struct describing system
      
      // IntergrationStep_ should be reset with the neuron on ResetNetwork,
      // but remain unchanged during calibration. Since it is initialized with
      // step_, and the resolution cannot change after nodes have been created,
      // it is safe to place both here.
      nest::double_t step_;           //!< step size in ms
      double   IntegrationStep_;//!< current integration time step, updated by GSL

      /** 
       * Input current injected by CurrentEvent.
       * This variable is used to transport the current applied into the
       * _dynamics function computing the derivative of the state vector.
       * It must be a part of Buffers_, since it is initialized once before
       * the first simulation, but not modified before later Simulate calls.
       */
      double_t I_stim_;
    };
    
    // Variables class ------------------------------------------------------- 
    
    /**
     * Internal variables of the model.
     * Variables are re-initialized upon each call to Simulate.
     */
    struct Variables_ { 
      /**
       * Impulse to add to DG_EXC on spike arrival to evoke unit-amplitude
       * conductance excursion.
       */
      nest::double_t PSConInit_E; 
      
      /**
       * Impulse to add to DG_INH on spike arrival to evoke unit-amplitude
       * conductance excursion.
       */
      nest::double_t PSConInit_I;    
      
      //! refractory time in steps
      nest::int_t    RefractoryCounts;
    };
    
    // Access functions for UniversalDataLogger -------------------------------
    
    //! Read out state vector elements, used by UniversalDataLogger
    template <State_::StateVecElems elem>
    nest::double_t get_y_elem_() const { return S_.y[elem]; }
    
    //! Read out remaining refractory time, used by UniversalDataLogger
    nest::double_t get_r_() const { return nest::Time::get_resolution().get_ms() * S_.r; }

    //! Read out Bias, used by UniversalDataLogger
    nest::double_t get_bias_() const { return S_.bias; }
    nest::double_t get_epsilon_() const { return S_.epsilon; }
    nest::double_t get_kappa_() const { return S_.kappa; }
    
    // Data members ----------------------------------------------------------- 

    // keep the order of these lines, seems to give best performance
    Parameters_ P_;
    State_      S_;
    Variables_  V_;
    Buffers_    B_;

    //! Mapping of recordables names to access functions
    static nest::RecordablesMap<iaf_cond_alpha_bias> recordablesMap_;
  };
  

  // Boilerplate inline function definitions ----------------------------------

  inline
  nest::port mynest::iaf_cond_alpha_bias::check_connection(nest::Connection& c, nest::port receptor_type)
  {
    nest::SpikeEvent e;
    e.set_sender(*this);
    c.check_event(e);
    return c.get_target()->connect_sender(e, receptor_type);
  }

  inline
  nest::port mynest::iaf_cond_alpha_bias::connect_sender(nest::SpikeEvent&, nest::port receptor_type)
  {
    if (receptor_type != 0)
      throw nest::UnknownReceptorType(receptor_type, get_name());
    return 0;
  }
 
  inline
  nest::port mynest::iaf_cond_alpha_bias::connect_sender(nest::CurrentEvent&, nest::port receptor_type)
  {
    if (receptor_type != 0)
      throw nest::UnknownReceptorType(receptor_type, get_name());
    return 0;
  }
 
  inline
  nest::port mynest::iaf_cond_alpha_bias::connect_sender(nest::DataLoggingRequest& dlr, 
				      nest::port receptor_type)
  {
    if (receptor_type != 0)
      throw nest::UnknownReceptorType(receptor_type, get_name());
    return B_.logger_.connect_logging_device(dlr, recordablesMap_);
  }

  inline
  void iaf_cond_alpha_bias::get_status(DictionaryDatum &d) const
  {
    P_.get(d);
    S_.get(d);
    nest::Archiving_Node::get_status(d);

    (*d)[nest::names::recordables] = recordablesMap_.get_list();
  }

  inline
  void iaf_cond_alpha_bias::set_status(const DictionaryDatum &d)
  {
    Parameters_ ptmp = P_;  // temporary copy in case of errors
    ptmp.set(d);                       // throws if BadProperty
    State_      stmp = S_;  // temporary copy in case of errors
    stmp.set(d, ptmp);                 // throws if BadProperty

    // We now know that (ptmp, stmp) are consistent. We do not 
    // write them back to (P_, S_) before we are also sure that 
    // the properties to be set in the parent class are internally 
    // consistent.
    nest::Archiving_Node::set_status(d);

    // if we get here, temporaries contain consistent set of properties
    P_ = ptmp;
    S_ = stmp;
  }

} // namespace

#endif //IAF_COND_ALPHA_BIAS_H

#endif //HAVE_GSL
