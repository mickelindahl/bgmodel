/*
 *  my_aeif_cond_exp.h
 *
 *  This file is part of NEST
 *
 *  Copyright (C) 2010 by
 *  The NEST Initiative
 *
 *  See the file AUTHORS for details.
 *
 *  Permission is granted to compile and modify
 *  this file for non-commercial use.
 *  See the file LICENSE for details.
 *  
 *  Synapes AMPA_1, NMDA_1, GABA_1 and GABA_2 added
 *  (C) 2011 Mikael Lindahl
 *
 */

#ifndef MY_AEIF_COND_EXP_H
#define MY_AEIF_COND_EXP_H

#include "config.h"


//Remeber, nest has to have been compiled with GSL before
//module is compiled.
#ifdef HAVE_GSL_1_11

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
Name: my_aeif_cond_exp - Conductance based exponential integrate-and-fire neuron model according to Brette and Gerstner (2005).

Description:
my_aeif_cond_exp is the adaptive exponential integrate and fire neuron according to Brette and Gerstner (2005).

This implementation uses the embedded 4th order Runge-Kutta-Fehlberg solver with adaptive stepsize to integrate
the differential equation.

The membrane potential is given by the following differential equation:
C dV/dt= -g_L(V-E_L)+g_L*Delta_T*exp((V-V_T)/Delta_T)-g_e(t)(V-E_e) -g_i(t)(V-E_i)-w +I_e

and

tau_w * dw/dt= a(V-E_L) -W

if V_a < V: a = a_1
else:       a = a_2

If V > V_peak
	if ( u < 0  )
		V = min(V_reset + u*V_reset_slope1, V_reset_max_slope1);
	else if ( u >= 0  )
		V = min(V_reset + u*V_reset_slope2, V_reset_max_slope1);

	u = u + a


I = g_AMPA_1( V - AMPA_1_E_rev) + g_NMDA_1( V - NMDA_1_E_rev) + g_GABAA( V - GABAA_E_rev) + I_e

with dopamine modulation

If I_AMPA_1=g_AMPA_1( V - AMPA_1_E_rev)
then I_AMPA_1=I_AMPA_1*(1-beta_I_AMPA_1*tata_dop).
Same for NMDA_1 and GABA.

Parameters: 
The following parameters can be set in the status dictionary.

Dynamic state variables:
  V_m        double - Membrane potential in mV
  w          double - Spike-adaptation current in pA.

Reset adaptation parameters:
V_reset_slope1          double - Slope of v below u=u_thr_slopes
V_reset_slope2          double - Slope of v above u=u_thr_slopes
V_reset_max_slope1      double - V max when u<u_thr_slopes
V_reset_max_slope2      double - V max when u>=u_thr_slopes

Membrane Parameters:
  C_m        double - Capacity of the membrane in pF
  t_ref      double - Duration of refractory period in ms. 
  V_peak     double - Spike detection threshold in mV.
  V_reset    double - Reset value for V_m after a spike. In mV.
  E_L        double - Leak reversal potential in mV. 
  g_L        double - Leak conductance in nS.
  I_e        double - Constant external input current in pA.

Spike adaptation parameters:
  V_a        double - Recovery variable voltage threshold
  a_1	       double - Subthreshold adaptation in nS. [-inf v_b]
  a_2	       double - Subthreshold adaptation in nS. [v_b +inf]
  b          double - Spike-triggered adaptation in pA.
  Delta_T    double - Slope factor in mV
  tau_w      double - Adaptation time constant in ms
  V_t        double - Spike initiation threshold in mV (V_th can also be used for compatibility).



Synaptic parameters
  AMPA_1_E_rev         double - AMPA_1 reversal potential in mV.
  AMPA_1_Tau_decay     double - Exponential decay time of the AMPA_1 synapse in ms.

  NMDA_1_E_rev         double - NMDA_1 reversal potential in mV.
  NMDA_1_Tau_decay     double - Exponential decay time of the NMDA_1 synaptse in ms.
  NMDA_1_Sact          double - For voltage dependence of NMDA_1-synapse mV, see eq. above
  NMDA_1_Vact          double - For voltage dependence of NMDA_1-synapse mV, see eq. ab

  GABAA_1_E_rev      double - GABAA 1 reversal potential in mV.
  GABAA_1_Tau_decay  double - Exponential decay time of the GABAA 1 synaptse in ms.

  GABAA_2_E_rev      double - GABAA 2 reversal potential in mV.
  GABAA_2_Tau_decay  double - Exponential decay time of the GABAA 2 synaptse in ms.

Integration parameters
  gsl_error_tol  double - This parameter controls the admissible error of the GSL integrator.
                          Reduce it if NEST complains about numerical instabilities.

Author: Adapted from aeif_cond_alpha by Lyle Muller

Sends: SpikeEvent

Receives: SpikeEvent, CurrentEvent, DataLoggingRequest

References: Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model as 
            an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642

SeeAlso: iaf_cond_exp, aeif_cond_alpha
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
int my_aeif_cond_exp_dynamics (double, const double*, double*, void*);

class my_aeif_cond_exp: public nest::Archiving_Node
{

public:        

	my_aeif_cond_exp();
	my_aeif_cond_exp(const my_aeif_cond_exp&);
	~my_aeif_cond_exp();

	/**
	 * Import sets of overloaded virtual functions.
	 * We need to explicitly include sets of overloaded
	 * virtual functions into the current scope.
	 * According to the SUN C++ FAQ, this is the correct
	 * way of doing things, although all other compilers
	 * happily live without.
	 */
	using nest::Node::connect_sender;
	using nest::Node::handle;

	nest::port check_connection(nest::Connection &, nest::port);

	void handle(nest::SpikeEvent &);
	void handle(nest::CurrentEvent &);
	void handle(nest::DataLoggingRequest &);

	nest::port connect_sender(nest::SpikeEvent &, nest::port);
	nest::port connect_sender(nest::CurrentEvent &, nest::port);
	nest::port connect_sender(nest::DataLoggingRequest &, nest::port);

	void get_status(DictionaryDatum &) const;
	void set_status(const DictionaryDatum &);

private:

	void init_node_(const Node &proto);
	void init_state_(const Node &proto);
	void init_buffers_();
	void calibrate();
	void update(const nest::Time &, const nest::long_t, const nest::long_t);


	/**
	 * Minimal spike receptor type.
	 * @note Start with 1 so we can forbid port 0 to avoid accidental
	 *       creation of connections with no receptor type set.
	 */
	static const nest::port MIN_SPIKE_RECEPTOR = 1;

	/**
	 * Spike receptors (SUP_SPIKE_RECEPTOR=3).
	 */
	// Three spike receptors AMPA_1, NMDA_1 and GABAA. OBS GABAA_3 is a dummy. Only not implemented yet.
	// Necessary to get the count right compared to izhik model
	enum SpikeSynapseTypes { AMPA_1=MIN_SPIKE_RECEPTOR, NMDA_1, GABAA_1, GABAA_2, AMPA_2,
		SUP_SPIKE_RECEPTOR };

	static const nest::size_t NUM_SPIKE_RECEPTORS = SUP_SPIKE_RECEPTOR - MIN_SPIKE_RECEPTOR;

	/**
	 * Minimal current receptor type.
	 *  @note Start with SUP_SPIKE_RECEPTOR to avoid any overlap and
	 *        accidental mix-ups.
	 */
	static const nest::port MIN_CURR_RECEPTOR = SUP_SPIKE_RECEPTOR;

	/**
	 * Current receptors (SUP_CURR_RECEPTOR = 5).
	 */
	enum CurrentSynapseTypes { CURR = MIN_CURR_RECEPTOR, SUP_CURR_RECEPTOR };

	static const nest::size_t NUM_CURR_RECEPTORS = SUP_CURR_RECEPTOR - MIN_CURR_RECEPTOR;


	// END Boilerplate function declarations ----------------------------

	// Friends --------------------------------------------------------

	// make dynamics function quasi-member
	friend int mynest::my_aeif_cond_exp_dynamics(double, const double*, double*, void*);

	// The next two classes need to be friends to access the State_ class/member
	friend class nest::RecordablesMap<my_aeif_cond_exp>;
	friend class nest::UniversalDataLogger<my_aeif_cond_exp>;

private:
	// ---------------------------------------------------------------- 

	//! Independent parameters
	struct Parameters_
	{
		double_t V_peak_;     //!< Spike detection threshold in mV
		double_t V_reset_;    //!< Reset Potential in mV
		double_t t_ref_;      //!< Refractory period in ms

		double_t g_L;         //!< Leak Conductance in nS
		double_t C_m;         //!< Membrane Capacitance in pF
		double_t E_L;         //!< Leak reversal Potential (aka resting potential) in mV
		double_t Delta_T;     //!< Slope faktor in ms.
		double_t tau_w;       //!< adaptation time-constant in ms.
		double_t b;           //!< Spike-triggered adaptation in pA
		double_t V_th;        //!< Spike threshold in mV.
		double_t I_e;         //!< Intrinsic current in pA.

		nest::double_t V_a;			   //!< Recovery variable voltage threshold
		nest::double_t a_1;				 //!< Subthreshold adaptation in nS. voltage interval [-inf v_b]
		nest::double_t a_2;				 //!< Subthreshold adaptation in nS. interval [v_b +inf]


		nest::double_t  V_reset_slope1;          //!< Slope of v rested point
		nest::double_t  V_reset_slope2;          //!< Slope of v rested point
		nest::double_t  V_reset_max_slope1;   //!< Max increase of v reset point
		nest::double_t  V_reset_max_slope2;   //!< Max increase of v reset point




		// Synaptic parameters
		nest::double_t AMPA_1_E_rev;        //!< AMPA_1 reversal Potential in mV
		nest::double_t AMPA_1_Tau_decay;    //!< Synaptic Time Constant AMPA_1 Synapse in ms

		nest::double_t AMPA_2_E_rev;        //!< AMPA_1 reversal Potential in mV
		nest::double_t AMPA_2_Tau_decay;    //!< Synaptic Time Constant AMPA_1 Synapse in ms

		nest::double_t NMDA_1_E_rev;        //!< NMDA_1 reversal Potential in mV
		nest::double_t NMDA_1_Tau_decay;    //!< Synaptic Time Constant NMDA_1 Synapse in ms
		nest::double_t NMDA_1_Vact;         //!< mV, inactive for V << Vact, inflection of sigmoid
		nest::double_t NMDA_1_Sact;         //!< mV, scale of inactivation

		nest::double_t GABAA_1_E_rev;    //!< GABAA 1 reversal Potential in mV
		nest::double_t GABAA_1_Tau_decay;//!< Rise Time Constant GABAA 1 Synapse in ms

		nest::double_t GABAA_2_E_rev;    //!< GABAA 2 reversal Potential in mV
		nest::double_t GABAA_2_Tau_decay;//!< Rise Time Constant GABAA 2 Synapse in ms

		double_t gsl_error_tol;   //!< error bound for GSL integrator

		// Dopamine modulation parameters
		nest::double_t tata_dop;       //!< Proportion of open dopamine receptors.

		nest::double_t beta_V_a;        //!< Dopamine effect on V_b
		nest::double_t beta_E_L;        //!< Dopamine effect on E_L


		//Synaptic dopamine modulation
		nest::double_t beta_I_AMPA_1;     //!< Dopamine effect on NMDA_1 current
		nest::double_t beta_I_AMPA_2;     //!< Dopamine effect on NMDA_1 current
		nest::double_t beta_I_NMDA_1;     //!< Dopamine effect on NMDA_1 current
		nest::double_t beta_I_GABAA_1;     //!< Dopamine effect GABAA 1 current
		nest::double_t beta_I_GABAA_2;     //!< Dopamine effect GABAA 2 current


		Parameters_();  //!< Sets default parameter values

		void get(DictionaryDatum &) const;  //!< Store current values in dictionary
		void set(const DictionaryDatum &);  //!< Set values from dicitonary
	};

public:
	// ---------------------------------------------------------------- 

	/**
	 * State variables of the model.
	 * @note Copy constructor and assignment operator required because
	 *       of C-style array.
	 */
	struct State_
	{

		//! Symbolic indices to the elements of the state vector y
		enum StateVecElems_ { V_M = 0, u,
			G_AMPA_1,
			G_AMPA_2,
			G_NMDA_1,
			G_GABAA_1,
			G_GABAA_2,
			STATE_VEC_SIZE };

		//! state vector, must be C-array for GSL solver
		nest::double_t y[STATE_VEC_SIZE];
		nest::int_t    r_;           //!< number of refractory steps remaining



		// Contructors, copy-constructors and destructors has to be defined in
		// .cpp in order to retrieve them.
		nest::double_t I_;  		   //!< Total input current from synapses and current injection
		nest::double_t I_AMPA_1_;    //!< AMPA_1 current; member only to allow recording
		nest::double_t I_AMPA_2_;    //!< AMPA_1 current; member only to allow recording
		nest::double_t I_NMDA_1_;    //!< NMDA_1 current; member only to allow recordin
		nest::double_t I_GABAA_1_; //!< GABAA current; member only to allow recording
		nest::double_t I_GABAA_2_; //!< GABAA current; member only to allow recording
		nest::double_t I_V_clamp_; //!< Current to inject in voltage clamp; member only to allow recording

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

		/**
		 * Enumeration identifying elements in state array State_::y_.
		 * The state vector must be passed to GSL as a C array. This enum
		 * identifies the elements of the vector. It must be public to be
		 * accessible from the iteration function.
		 */  

	};

	// ---------------------------------------------------------------- 

	/**
	 * Buffers of the model.
	 */
	struct Buffers_
	{
		Buffers_(my_aeif_cond_exp &);                    //!<Sets buffer pointers to 0
		Buffers_(const Buffers_ &, my_aeif_cond_exp &);  //!<Sets buffer pointers to 0

		//! Logger for all analog data
		nest::UniversalDataLogger<my_aeif_cond_exp> logger_;

		/** buffers and sums up incoming spikes/currents */
		// One ring buffer for each synapse. Need to register this in receptor
		// dictionary.
		nest::RingBuffer spikes_AMPA_1_;
		nest::RingBuffer spikes_AMPA_2_;
		nest::RingBuffer spikes_NMDA_1_;
		nest::RingBuffer spikes_GABAA_1_;
		nest::RingBuffer spikes_GABAA_2_;
		nest::RingBuffer currents_;


		/** GSL ODE stuff */
		gsl_odeiv_step*    s_;    //!< stepping function
		gsl_odeiv_control* c_;    //!< adaptive stepsize control function
		gsl_odeiv_evolve*  e_;    //!< evolution function
		gsl_odeiv_system   sys_;  //!< struct describing system

		// IntergrationStep_ should be reset with the neuron on ResetNetwork,
		// but remain unchanged during calibration. Since it is initialized with
		// step_, and the resolution cannot change after nodes have been created,
		// it is safe to place both here.
		nest::double_t step_;             //!< step size in ms
		double   IntegrationStep_;  //!< current integration time step, updated by GSL

		/** 
		 * Input current injected by CurrentEvent.
		 * This variable is used to transport the current applied into the
		 * _dynamics function computing the derivative of the state vector.
		 * It must be a part of Buffers_, since it is initialized once before
		 * the first simulation, but not modified before later Simulate calls.
		 */
		nest::double_t I_stim_;
	};

	// ---------------------------------------------------------------- 

	/**
	 * Internal variables of the model.

	 */
	struct Variables_
	{
		/** Impulse to add to DG_AMPA_1 on spike arrival to evoke unit-amplitude
		 * conductance excursion.
		 */
		//nest::double_t PSConInit_AMPA_1;

		/**
		 * Impulse to add to DG_NMDA_1 on spike arrival to evoke unit-amplitude
		 * conductance excursion.
		 */
		//nest::double_t PSConInit_NMDA_1;

		/**
		 * Impulse to add to DG_GABAA on spike arrival to evoke unit-amplitude
		 * conductance excursion.
		 */
		//nest::double_t PSConInit_GABAA;

		nest::int_t RefractoryCounts_;
	};

	// Access functions for UniversalDataLogger -------------------------------

	//! Read out state vector elements, used by UniversalDataLogger
	template <State_::StateVecElems_ elem>
	nest::double_t get_y_elem_()    const { return S_.y[elem]; }
	nest::double_t get_I_() 				const { return S_.I_; }
	nest::double_t get_I_AMPA_1_() 		const { return S_.I_AMPA_1_; 		}
	nest::double_t get_I_AMPA_2_() 		const { return S_.I_AMPA_2_; 		}
	nest::double_t get_I_NMDA_1_() 		const { return S_.I_NMDA_1_;    }
	nest::double_t get_I_GABAA_1_() const { return S_.I_GABAA_1_; }
	nest::double_t get_I_GABAA_2_() const { return S_.I_GABAA_2_; }
	nest::double_t get_I_V_clamp_() const { return S_.I_V_clamp_; }

	// ---------------------------------------------------------------- 

	Parameters_ P_;
	State_      S_;
	Variables_  V_;
	Buffers_    B_;

	//! Mapping of recordables names to access functions
	static nest::RecordablesMap<my_aeif_cond_exp> recordablesMap_;
};

inline
nest::port mynest::my_aeif_cond_exp::check_connection(nest::Connection& c, nest::port receptor_type)
{
	nest::SpikeEvent e;
	e.set_sender(*this);
	c.check_event(e);
	return c.get_target()->connect_sender(e, receptor_type);
}

inline
nest::port mynest::my_aeif_cond_exp::connect_sender(nest::SpikeEvent&, nest::port receptor_type)
{
	// If receptor type is less than 1 =(MIN_SPIKE_RECEPTOR) or greater or equal to 4
	// (=SUP_SPIKE_RECEPTOR) then provided receptor type is not a spike receptor.
	if ( receptor_type < MIN_SPIKE_RECEPTOR || receptor_type >= SUP_SPIKE_RECEPTOR )
		// Unknown receptor type is less than 0 or greater than 6
		// (SUP_CURR_RECEPTOR).
		if ( receptor_type < 0 || receptor_type >= SUP_CURR_RECEPTOR )
			throw nest::UnknownReceptorType(receptor_type, get_name());
	// Otherwise it is a current receptor or receptor 0 (data logging request
	// not used here and therefore incompatible.
		else
			throw nest::IncompatibleReceptorType(receptor_type, get_name(), "SpikeEvent");
	// If we arrive here the receptor type is a spike receptor and either 1, 2 or 3 e.i.
	// greater or equal to MIN_SPIKE_RECEPTOR = 1, and less than SUP_SPIKE_RECEPTOR
	// = 4. Then 0, 1, or 2 is returned.
	return receptor_type - MIN_SPIKE_RECEPTOR;
}

inline
nest::port mynest::my_aeif_cond_exp::connect_sender(nest::CurrentEvent&, nest::port receptor_type)
{
	// If receptor type is less than 4 (MIN_CURR_RECEPTOR) or greater or equal
	// to 5 (SUP_CURR_RECEPTOR) the provided receptor type is not current
	// receptor.
	if ( receptor_type < MIN_CURR_RECEPTOR || receptor_type >= SUP_CURR_RECEPTOR )
		// If receptor is not a current receptor but still a receptor type that is
		// the receptor type is greater or equal to 0 or less than 3
		// (MIN_CURR_RECEPTOR).
		if ( receptor_type >= 0 && receptor_type < MIN_CURR_RECEPTOR )
			throw nest::IncompatibleReceptorType(receptor_type, get_name(), "CurrentEvent");
	// Otherwise unknown receptor type.
		else
			throw nest::UnknownReceptorType(receptor_type, get_name());
	//MIN_CURR_RECEPTOR =4, If here receptor type equals 4  and 0 is returned.
	return receptor_type - MIN_CURR_RECEPTOR;
}

inline
nest::port mynest::my_aeif_cond_exp::connect_sender(nest::DataLoggingRequest& dlr,
		nest::port receptor_type)
{
	// If receptor type does not equal 0 then it is not a data logging request
	// receptor.
	if ( receptor_type != 0 )
		// If not a spike or current receptor that is less than 0 or greater or
		//  equal to 4 (SUP_CURR_RECEPTOR).
		if ( receptor_type < 0 || receptor_type >= SUP_CURR_RECEPTOR )
			throw nest::UnknownReceptorType(receptor_type, get_name());
	// Otherwise it is a spike or current receptor type.
		else
			throw nest::IncompatibleReceptorType(receptor_type, get_name(), "DataLoggingRequest");
	// CHANGED
	//B_.logger_.connect_logging_device(dlr, recordablesMap_);
	//return 0;

	// TO
	return B_.logger_.connect_logging_device(dlr, recordablesMap_);

}




inline
void my_aeif_cond_exp::get_status(DictionaryDatum &d) const
{
	P_.get(d);
	S_.get(d);
	nest::Archiving_Node::get_status(d);


	(*d)[nest::names::recordables] = recordablesMap_.get_list();
	/**
	 * @TODO dictionary construction should be done only once for
	 * static member in default c'tor, but this leads to
	 * a seg fault on exit, see #328
	 */
	DictionaryDatum receptor_dict_ = new Dictionary();
	(*receptor_dict_)[Name("AMPA_1")]    = AMPA_1;
	(*receptor_dict_)[Name("AMPA_2")]    = AMPA_2;
	(*receptor_dict_)[Name("NMDA_1")]    = NMDA_1;
	(*receptor_dict_)[Name("GABAA_1")] = GABAA_1;
	(*receptor_dict_)[Name("GABAA_2")] = GABAA_2;
	(*receptor_dict_)[Name("CURR")]    = CURR;

	(*d)[nest::names::receptor_types] = receptor_dict_;
}

inline
void my_aeif_cond_exp::set_status(const DictionaryDatum &d)
{
	Parameters_ ptmp = P_;  // temporary copy in case of errors
	ptmp.set(d);            // throws if BadProperty
	State_      stmp = S_;  // temporary copy in case of errors
	stmp.set(d, ptmp);      // throws if BadProperty

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

#endif // HAVE_GSL_1_11
#endif // MY_AEIF_COND_EXP_H
