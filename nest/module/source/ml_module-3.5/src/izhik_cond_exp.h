/*
 *  izhik_cond_exp.h
 *
 *  This file is part of ML_MODULE
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
 *   Izhikivich simple model described in Dynamical Systems In Neuroscience
 *   2007
 *  (C) 2010 Mikael Lindahl
 *
 */

#ifndef IZHIK_COND_EXP_H
#define IZHIK_COND_EXP_H

// Generated includes:
#include "config.h"

//Remeber, nest has to have been compiled with GSL before
//module is compiled.
#ifdef HAVE_GSL

// External includes:
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

//#include "nest.h"

// Includes from nestkernel:
#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "recordables_map.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

/* BeginDocumentation
Name: izhik_cond_exp - Izhikevich simple neuron

Description:
izhik_cond_exp is an implementation of based on  Izhikevich simple model
presented in (Izhikevich 2007). The model has two equation one is the voltage
and the other is a slow recovery variable and exponential shaped synaptic conductancies.


Model equations:
C*dV/dt = k( V - E_L )( v - V_th ) - u - I
du/dt = a( b( V - E_L )^p - u + u_const )

with dopamine modulation as (with tata_dop=0 is as without):
k   = k*(1-beta_k*tata_dop)
V_b = V_b*(1-beta_V_b*tata_dop)
E_L = E_L*(1-beta_E_L*tata_dop)
d   = d*(1-beta_E_L*tata_dop)

I = g_AMPA_1( V - AMPA_1_E_rev) + g_NMDA_1( V - NMDA_1_E_rev) + g_GABAA( V - GABAA_E_rev) + I_e


with dopamine modulation

If I_AMPA_1=g_AMPA_1( V - AMPA_1_E_rev)
then I_AMPA_1=I_AMPA_1*(1-beta_I_AMPA_1*tata_dop).
Same for NMDA_1 and GABA.

if V_b < V: b = b_1 and p = p_1
else:       b = b_2 and p = p_2

If V > V_peak
	if ( u < P_.u_kc  )
		V = P_.c + u*kc_1;
	else
		V = P_.c + u*kc_1;

	if V > c_max
		V = c_max
		u = u + d
	if u > P.u_max )
		u = P.u_max;


Parameters: 
The following parameters can be set in the status dictionary.

Experiment parameters
V_clamp		 double - 0 or 1, no/yes for voltage clamp, i.e. when 1 then membrane
										voltage will be clamped at V_clamp_at
V_clamp_at double - Membrane potential in mV

Neuron parameters:
Voltage equation parameters
V_m        double - Membrane potential in mV 
E_L        double - Membrane resting potential
V_th			   double - Instantaneous activation threshold
C_m        double - Capacity of the membrane in pF
k					 double - factor determining instantaneous I-V relation
c					 double - Reset value membrane potential
kc_1       double - Slope of voltage reset point when u < u_kc with respect to u
kc_2 		   double - Slope of voltage reset point when u > u_kc with respect to u
u_kc;			 double - When u < u_kc, kc = kc_1 and when u > u_kc, kc = kc_2
V_peak     double - spike cut of value
I_e        double - Constant input current in pA.

Recovery variable equations
u          double - Recovery variable
V_b        double - Recovery variable voltage threshold
a	         double - Time constant slow dynamics
b_1	       double - Slope factor slow dynamics voltage interval [-inf v_b]
b_2	       double - Slope factor slow dynamics interval [v_b +inf]
p_1 	     double - Recovery variable polynomial voltage dependency factor  interval  [-inf v_b]
p_2 	     double - Recovery variable polynomial voltage dependency factor  interval  [v_b +inf]
d  	   	   double - slow variable change at spike
u_const    double - Constant input current in pA to recovery variable u.
u_max      double - maximum value that u can take
c_max	     double - maximum value that c can take

Synapse parameters

AMPA_1_E_rev         double - AMPA_1 reversal potential in mV.
AMPA_1_Tau_decay     double - Exponential decay time of the AMPA_1 synapse in ms.

NMDA_1_E_rev         double - NMDA_1 reversal potential in mV.
NMDA_1_Tau_decay     double - Exponential decay time of the NMDA_1 synapse in ms.
NMDA_1_Sact          double - For voltage dependence of NMDA_1-synapse mV, see eq. above
NMDA_1_Vact          double - For voltage dependence of NMDA_1-synapse mV, see eq. ab

GABAA_1_E_rev      double - GABAA 1 reversal potential in mV.
GABAA_1_Tau_decay  double - Exponential decay time of the GABAA 1 synapse in ms.

GABAA_2_E_rev      double - GABAA 2 reversal potential in mV.
GABAA_2_Tau_decay  double - Exponential decay time of the GABAA 2 synapse in ms.

GABAA_3_E_rev      double - GABAA 3 reversal potential in mV.
GABAA_3_Tau_decay  double - Exponential decay time of the GABAA 3 synapse in ms.

Sends: SpikeEvent

Receives: SpikeEvent, CurrentEvent, DataLoggingRequest

Author: Mikael, Lindahl

SeeAlso: iaf_cond_exp, iaf_cond_alpha, iaf_cond_alpha_mc, ht_neuron
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
int izhik_cond_exp_dynamics (double, const double*, double*, void*);

/**
 * Integrate-and-fire neuron model with two conductance-based synapses.
 *
 * @note Per 2009-04-17, this class has been revised to our newest
 *       insights into class design. Please use THIS CLASS as a reference
 *       when designing your own models with nonlinear dynamics.
 *       One weakness of this class is that it distinguishes between
 *       inputs to the two synapses by the sign of the synaptic weight.
 *       It would be better to use receptor_types, cf izhik_cond_exp_mc.
 */
class izhik_cond_exp : public nest::ArchivingNode
{

	// Boilerplate function declarations --------------------------------

public:

	izhik_cond_exp();
	izhik_cond_exp(const izhik_cond_exp&);
	~izhik_cond_exp();

    /**
     * Import sets of overloaded virtual functions.
     * @see Technical Issues / Virtual Functions: Overriding, Overloading, and Hiding
     */
	using nest::Node::handle;
    using nest::Node::handles_test_event;

    size_t send_test_event(nest::Node&, size_t, nest::synindex, bool) override;

	void handle(nest::SpikeEvent &) override;
	void handle(nest::CurrentEvent &) override;
	void handle(nest::DataLoggingRequest &) override;

   size_t handles_test_event(nest::SpikeEvent &,size_t) override;
   size_t handles_test_event(nest::CurrentEvent &,size_t) override;
   size_t handles_test_event(nest::DataLoggingRequest &,size_t) override;

	void get_status(DictionaryDatum &) const override;
	void set_status(const DictionaryDatum &)  override;

private:
	void init_buffers_()  override;
    void pre_run_hook() override;
	void update(nest::Time const &, const long, const long) override;


	/**
	 * Minimal spike receptor type.
	 * @note Start with 1 so we can forbid port 0 to avoid accidental
	 *       creation of connections with no receptor type set.
	 */
	static const int MIN_SPIKE_RECEPTOR = 1;

	/**
	 * Spike receptors (SUP_SPIKE_RECEPTOR=3).
	 */
	//  OBS AMPA_2 is a dummy. Only not implemented yet.
	enum SpikeSynapseTypes { AMPA_1=MIN_SPIKE_RECEPTOR, NMDA_1, GABAA_1, GABAA_2, GABAA_3, AMPA_2,
		SUP_SPIKE_RECEPTOR };

	static const int NUM_SPIKE_RECEPTORS = SUP_SPIKE_RECEPTOR - MIN_SPIKE_RECEPTOR;

	/**
	 * Minimal current receptor type.
	 *  @note Start with SUP_SPIKE_RECEPTOR to avoid any overlap and
	 *        accidental mix-ups.
	 */
	static const int MIN_CURR_RECEPTOR = SUP_SPIKE_RECEPTOR;

	/**
	 * Current receptors (SUP_CURR_RECEPTOR = 5).
	 */
	enum CurrentSynapseTypes { CURR = MIN_CURR_RECEPTOR, SUP_CURR_RECEPTOR };

	static const int NUM_CURR_RECEPTORS = SUP_CURR_RECEPTOR - MIN_CURR_RECEPTOR;


	// Friends --------------------------------------------------------

	// make dynamics function quasi-member. Add mynest since this is a function
	//in namespace mynest.
	friend int mynest::izhik_cond_exp_dynamics(double, const double*, double*, void*);


	// The next two classes need to be friends to access the State_ class/member
	friend class nest::RecordablesMap<izhik_cond_exp>;
	friend class nest::UniversalDataLogger<izhik_cond_exp>;

private:

	// Parameters class -------------------------------------------------

	//! Model parameters
	struct Parameters_ {

		// Experimental parameters
		double V_clamp;    //!< 0 or 1, no/yes for voltage clamp, i.e. when 1 then membrane voltage will be clamped at V_clamp_at
		double V_clamp_at; //!< Membrane potential in mV

		// Voltage equation parameters
		double V_m; 			 //!< Membrane potential in mV
		double E_L;        //!< Membrane resting potential
		double V_th;			   //!< Instantaneous activation threshold
		double C_m;        //!< Capacity of the membrane in pF
		double k;					 //!< Factor determining instantaneous I-V relation
		double c;					 //!< Reset value membrane potential
		double kc_1;       //!< Slope of voltage reset point when u < u_kc with respect to u
		double kc_2;    	 //!< Slope of voltage reset point when u > u_kc with respect to u
		double u_kc;		   //!< When u < u_kc, kc = kc_1 and when u > u_kc, kc = kc_2
		double V_peak;     //!< spike cut of value
		double I_e;        //!< Constant input current in pA.



		// Recovery variable equations
		double u; 				 //!< Recovery variable
		double V_b;			   //!< Recovery variable voltage threshold
		double a;			     //!< Time constant slow dynamics
		double b_1;				 //!< Slope factor slow dynamics voltage interval [-inf v_b]
		double b_2;				 //!< Slope factor slow dynamics interval [v_b +inf]
		double p_1;				 //!< Recovery variable polynomial voltage dependency factor  interval  [-inf v_b]
		double p_2;				 //!< Recovery variable voltage dependency factor interval 1 [v_b +inf]
		double d;			     //!< Slow variable change at spike
		double u_const;    //!< Constant input current in pA to recovery variable u.
		double u_max;      //!< maximum value that u can take
		double c_max;			 //!< maximum value that c can take

		// Synaptic parameters
		double AMPA_1_E_rev;        //!< AMPA_1 reversal Potential in mV
		double AMPA_1_Tau_decay;    //!< Synaptic Time Constant AMPA_1 Synapse in ms

		double NMDA_1_E_rev;        //!< NMDA_1 reversal Potential in mV
		double NMDA_1_Tau_decay;    //!< Synaptic Time Constant NMDA_1 Synapse in ms
		double NMDA_1_Vact;         //!< mV, inactive for V << Vact, inflection of sigmoid
		double NMDA_1_Sact;         //!< mV, scale of inactivation

		double GABAA_1_E_rev;    //!< GABAA 1 reversal Potential in mV
		double GABAA_1_Tau_decay;//!< Rise Time Constant GABAA 1 Synapse in ms

		double GABAA_2_E_rev;    //!< GABAA 2 reversal Potential in mV
		double GABAA_2_Tau_decay;//!< Rise Time Constant GABAA 2 Synapse in ms

		double GABAA_3_E_rev;    //!< GABAA 3 reversal Potential in mV
		double GABAA_3_Tau_decay;//!< Rise Time Constant GABAA 2 Synapse in ms

		// Dopamine modulation parameters
		double tata_dop;       //!< Proportion of open dopamine receptors.

		// Neuron model dopamine modulation
		double beta_d;     //!< Dopamine effect on d
		double beta_k;     //!< Dopamine effect on k
		double beta_V_b;        //!< Dopamine effect on V_b
		double beta_E_L;        //!< Dopamine effect on E_L


		//Synaptic dopamine modulation
		double beta_I_AMPA_1;     //!< Dopamine effect on NMDA_1 current
		double beta_I_NMDA_1;     //!< Dopamine effect on NMDA_1 current
		double beta_I_GABAA_1;     //!< Dopamine effect GABAA 1 current
		double beta_I_GABAA_2;     //!< Dopamine effect GABAA 2 current
		double beta_I_GABAA_3;     //!< Dopamine effect GABAA 3 current

		Parameters_();                    //!< Set default parameter values

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
	struct State_ {

		//! Symbolic indices to the elements of the state vector y
		enum StateVecElems_ { V_M = 0, u,
			G_AMPA_1,
			G_NMDA_1,
			G_GABAA_1,
			G_GABAA_2,
			G_GABAA_3,
			STATE_VEC_SIZE };

		//! state vector, must be C-array for GSL solver
		double y[STATE_VEC_SIZE];

		//!< number of refractory steps remaining
		//unsigned int     r;

		// Contructors, copy-constructors and destructors has to be defined in
		// .cpp in order to retrieve them.
		double I_;  		   //!< Total input current from synapses and current injection
		double I_AMPA_1_;    //!< AMPA_1 current; member only to allow recording
		double I_NMDA_1_;    //!< NMDA_1 current; member only to allow recording
		double I_GABAA_;   //!< GABAA current; member only to allow recording
		double I_GABAA_1_; //!< GABAA current; member only to allow recording
		double I_GABAA_2_; //!< GABAA current; member only to allow recording
		double I_GABAA_3_; //!< GABAA current; member only to allow recording
		double I_V_clamp_; //!< Current to inject in voltage clamp; member only to allow recording

		// Taken from ht_neuron use to be only State_(const Parameters_&) and
		// State_(const State_&). Changed since compiler complained that I_AMPA_1,
		// I_NMDA_1 and I_GABAA read-only structure
		State_();
		State_(const Parameters_& p);
		State_(const State_& s);
		~State_();

		State_& operator=(const State_& s);

		void get(DictionaryDatum&) const;  //!< Store current values in dictionary

		/**
		 * Set state from values in dictionary.
		 * Requires Parameters_ as argument to, eg, check bounds.'
		 */
		void set(const DictionaryDatum&, const Parameters_&);


	};

	// Buffers class --------------------------------------------------------

	/**
	 * Buffers of the model.
	 * Buffers are on par with state variables in terms of persistence,
	 * i.e., initalized only upon first Simulate call after ResetKernel
	 * or ResetNetwork, but are implementation details hidden from the user.
	 */
	struct Buffers_ {
		Buffers_(izhik_cond_exp&); //!<Sets buffer pointers to 0
		Buffers_(const Buffers_&, izhik_cond_exp&); //!<Sets buffer pointers to 0

		//! Logger for all analog data
		nest::UniversalDataLogger<izhik_cond_exp> logger_;

		/** buffers and sums up incoming spikes/currents */
		// One ring buffer for each synapse. Need to register this in receptor
		// dictionary.
		nest::RingBuffer spikes_AMPA_1_;
		nest::RingBuffer spikes_NMDA_1_;
		nest::RingBuffer spikes_GABAA_1_;
		nest::RingBuffer spikes_GABAA_2_;
		nest::RingBuffer spikes_GABAA_3_;
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
		double step_;           //!< step size in ms
		double   IntegrationStep_;//!< current integration time step, updated by GSL

		/**
		 * Input current injected by CurrentEvent.
		 * This variable is used to transport the current applied into the
		 * _dynamics function computing the derivative of the state vector.
		 * It must be a part of Buffers_, since it is initialized once before
		 * the first simulation, but not modified before later Simulate calls.
		 */
		double I_stim_;
	};

	// Variables class -------------------------------------------------------

	/**
	 * Internal variables of the model.
	 * Variables are re-initialized upon each call to Simulate.
	 */
	struct Variables_ {
	};

	// Access functions for UniversalDataLogger -------------------------------

	//! Read out state vector elements and currents, used by UniversalDataLogger
	template <State_::StateVecElems_ elem>
	double get_y_elem_() 		const { return S_.y[elem]; 		}
	double get_I_() 				const { return S_.I_; }
	double get_I_AMPA_1_() 		const { return S_.I_AMPA_1_; 		}
	double get_I_NMDA_1_() 		const { return S_.I_NMDA_1_;    }
	double get_I_GABAA_1_() const { return S_.I_GABAA_1_; }
	double get_I_GABAA_2_() const { return S_.I_GABAA_2_; }
	double get_I_GABAA_3_() const { return S_.I_GABAA_3_; }
	double get_I_V_clamp_() const { return S_.I_V_clamp_; }

	//! Read out remaining refractory time, used by UniversalDataLogger
	//double get_r_() const { return nest::Time::get_resolution().get_ms() * S_.r; }

	// Data members -----------------------------------------------------------

	// keep the order of these lines, seems to give best performance
	Parameters_ P_;
	State_      S_;
	Variables_  V_;
	Buffers_    B_;

	//! Mapping of recordables names to access functions
	static nest::RecordablesMap<izhik_cond_exp> recordablesMap_;
};


// Boilerplate inline function definitions ----------------------------------

inline size_t
mynest::izhik_cond_exp::send_test_event(nest::Node& target, size_t receptor_type, nest::synindex, bool)
{
  nest::SpikeEvent e;
  e.set_sender(*this);
  return target.handles_test_event(e, receptor_type);
}

inline size_t
mynest::izhik_cond_exp::handles_test_event(nest::SpikeEvent&, size_t receptor_type)
{
  if ( receptor_type < MIN_SPIKE_RECEPTOR || receptor_type >= SUP_SPIKE_RECEPTOR )
  {
    if ( receptor_type < 0 || receptor_type >= SUP_CURR_RECEPTOR )
	throw nest::UnknownReceptorType(receptor_type, get_name());
    else
	throw nest::IncompatibleReceptorType(receptor_type, get_name(), "SpikeEvent");
  }
  return receptor_type - MIN_SPIKE_RECEPTOR;
}

inline size_t
mynest::izhik_cond_exp::handles_test_event(nest::CurrentEvent&, size_t receptor_type)
{
  if ( receptor_type < MIN_CURR_RECEPTOR || receptor_type >= SUP_CURR_RECEPTOR )
  {
    if ( receptor_type >= 0 && receptor_type < MIN_CURR_RECEPTOR )
	throw nest::IncompatibleReceptorType(receptor_type, get_name(), "CurrentEvent");
    else
	throw nest::UnknownReceptorType(receptor_type, get_name());
  }
  return receptor_type - MIN_CURR_RECEPTOR;
}

inline size_t
mynest::izhik_cond_exp::handles_test_event(nest::DataLoggingRequest& dlr,
		size_t receptor_type)
{
  if ( receptor_type != 0 )
  {
    if ( receptor_type < 0 || receptor_type >= SUP_CURR_RECEPTOR )
	throw nest::UnknownReceptorType(receptor_type, get_name());
    else
	throw nest::IncompatibleReceptorType(receptor_type, get_name(), "DataLoggingRequest");
  }
  return B_.logger_.connect_logging_device(dlr, recordablesMap_);
}

inline void
izhik_cond_exp::get_status(DictionaryDatum &d) const
{
	P_.get(d);
	S_.get(d);
	nest::ArchivingNode::get_status(d);

	(*d)[nest::names::recordables] = recordablesMap_.get_list();

	/**
	 * @TODO dictionary construction should be done only once for
	 * static member in default c'tor, but this leads to
	 * a seg fault on exit, see #328
	 */
	DictionaryDatum receptor_dict_ = new Dictionary();
	(*receptor_dict_)[Name("AMPA_1")]    = AMPA_1;
	(*receptor_dict_)[Name("NMDA_1")]    = NMDA_1;
	(*receptor_dict_)[Name("GABAA_1")] = GABAA_1;
	(*receptor_dict_)[Name("GABAA_2")] = GABAA_2;
	(*receptor_dict_)[Name("GABAA_3")] = GABAA_3;
	(*receptor_dict_)[Name("CURR")]    = CURR;

	(*d)[nest::names::receptor_types] = receptor_dict_;

}

inline
void izhik_cond_exp::set_status(const DictionaryDatum &d)
{
	Parameters_ ptmp = P_;  // temporary copy in case of errors
	ptmp.set(d);                       // throws if BadProperty
	State_      stmp = S_;  // temporary copy in case of errors
	stmp.set(d, ptmp);                 // throws if BadProperty

	// We now know that (ptmp, stmp) are consistent. We do not
	// write them back to (P_, S_) before we are also sure that
	// the properties to be set in the parent class are internally
	// consistent.
	nest::ArchivingNode::set_status(d);

	// if we get here, temporaries contain consistent set of properties
	P_ = ptmp;
	S_ = stmp;
}

} // namespace


#endif //HAVE_GSL
#endif //IZHIK_COND_EXP_H
