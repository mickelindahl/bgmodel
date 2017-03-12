/*
 *  my_aeif_cond_exp.h
 *
 *  This file is part of NEST
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
 *  Synapes AMPA_1, NMDA_1, GABA_1 and GABA_2 added
 *  (C) 2011 Mikael Lindahl
 *
 */

#ifndef MY_AEIF_COND_EXP_H
#define MY_AEIF_COND_EXP_H

// Generated includes:
#include "config.h"

//Remeber, nest has to have been compiled with GSL before
//module is compiled.
#ifdef HAVE_GSL //HAVE_GSL_1_11

// External includes:
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

// Includes from nestkernel:
#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "recordables_map.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

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
     * @see Technical Issues / Virtual Functions: Overriding, Overloading, and Hiding
     */
//	using nest::Node::connect_sender;
	using nest::Node::handle;
    using Node::handles_test_event;

    nest::port send_test_event(Node&, nest::rport, nest::synindex, bool);

	void handle(nest::SpikeEvent &);
	void handle(nest::CurrentEvent &);
	void handle(nest::DataLoggingRequest &);

    nest::port handles_test_event(nest::SpikeEvent &, nest::rport);
    nest::port handles_test_event(nest::CurrentEvent &, nest::rport);
    nest::port handles_test_event(nest::DataLoggingRequest &, nest::rport);

	void get_status(DictionaryDatum &) const;
	void set_status(const DictionaryDatum &);

private:

	void init_node_(const Node &proto);
	void init_state_(const Node &proto);
	void init_buffers_();
	void calibrate();
	void update(const nest::Time &, const long, const long);


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

	static const int NUM_SPIKE_RECEPTORS = SUP_SPIKE_RECEPTOR - MIN_SPIKE_RECEPTOR;

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

	static const int NUM_CURR_RECEPTORS = SUP_CURR_RECEPTOR - MIN_CURR_RECEPTOR;


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
		double C_m;         //!< Membrane Capacitance in pF
		double V_th;        //!< Spike threshold in mV.
		double t_ref_;      //!< Refractory period in ms

	    double V_reset_;    //!< Reset Potential in mV

	    double g_L;         //!< Leak Conductance in nS
		double E_L;         //!< Leak reversal Potential (aka resting potential) in mV

    	double a_1;				 //!< Subthreshold adaptation in nS. voltage interval [-inf v_b]
		double a_2;				 //!< Subthreshold adaptation in nS. interval [v_b +inf]
    	double V_a;			   //!< Recovery variable voltage threshold

		double b;           //!< Spike-triggered adaptation in pA
		double Delta_T;     //!< Slope faktor in ms.
		double tau_w;       //!< adaptation time-constant in ms.
		double I_e;         //!< Intrinsic current in pA.
	    double V_peak_;     //!< Spike detection threshold in mV


		double  V_reset_slope1;          //!< Slope of v rested point
		double  V_reset_slope2;          //!< Slope of v rested point
		double  V_reset_max_slope1;   //!< Max increase of v reset point
		double  V_reset_max_slope2;   //!< Max increase of v reset point

		// Synaptic parameters
		double AMPA_1_E_rev;        //!< AMPA_1 reversal Potential in mV
		double AMPA_1_Tau_decay;    //!< Synaptic Time Constant AMPA_1 Synapse in ms

		double AMPA_2_E_rev;        //!< AMPA_1 reversal Potential in mV
		double AMPA_2_Tau_decay;    //!< Synaptic Time Constant AMPA_1 Synapse in ms

		double NMDA_1_E_rev;        //!< NMDA_1 reversal Potential in mV
		double NMDA_1_Tau_decay;    //!< Synaptic Time Constant NMDA_1 Synapse in ms
		double NMDA_1_Vact;         //!< mV, inactive for V << Vact, inflection of sigmoid
		double NMDA_1_Sact;         //!< mV, scale of inactivation

		double GABAA_1_E_rev;    //!< GABAA 1 reversal Potential in mV
		double GABAA_1_Tau_decay;//!< Rise Time Constant GABAA 1 Synapse in ms

		double GABAA_2_E_rev;    //!< GABAA 2 reversal Potential in mV
		double GABAA_2_Tau_decay;//!< Rise Time Constant GABAA 2 Synapse in ms

		double gsl_error_tol;   //!< error bound for GSL integrator

		// Dopamine modulation parameters
		double tata_dop;       //!< Proportion of open dopamine receptors.

		double beta_V_a;        //!< Dopamine effect on V_b
		double beta_E_L;        //!< Dopamine effect on E_L


		//Synaptic dopamine modulation
		double beta_I_AMPA_1;     //!< Dopamine effect on NMDA_1 current
		double beta_I_AMPA_2;     //!< Dopamine effect on NMDA_1 current
		double beta_I_NMDA_1;     //!< Dopamine effect on NMDA_1 current
		double beta_I_GABAA_1;     //!< Dopamine effect GABAA 1 current
		double beta_I_GABAA_2;     //!< Dopamine effect GABAA 2 current


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


		// Contructors, copy-constructors and destructors has to be defined in
		// .cpp in order to retrieve them.
		double I_;  		   //!< Total input current from synapses and current injection
		double I_AMPA_1_;    //!< AMPA_1 current; member only to allow recording
		double I_AMPA_2_;    //!< AMPA_1 current; member only to allow recording
		double I_NMDA_1_;    //!< NMDA_1 current; member only to allow recordin
		double I_GABAA_1_; //!< GABAA current; member only to allow recording
		double I_GABAA_2_; //!< GABAA current; member only to allow recording
		double I_V_clamp_; //!< Current to inject in voltage clamp; member only to allow recording

		//! Symbolic indices to the elements of the state vector y
		enum StateVecElems_ { V_M = 0, u,
			G_AMPA_1,
			G_AMPA_2,
			G_NMDA_1,
			G_GABAA_1,
			G_GABAA_2,
			STATE_VEC_SIZE };


    	//! state vector, must be C-array for GSL solver
		double y[STATE_VEC_SIZE];
        unsigned int     r_;           //!< number of refractory steps remaining


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
		double step_;             //!< step size in ms
		double   IntegrationStep_;  //!< current integration time step, updated by GSL

		/** 
		 * Input current injected by CurrentEvent.
		 * This variable is used to transport the current applied into the
		 * _dynamics function computing the derivative of the state vector.
		 * It must be a part of Buffers_, since it is initialized once before
		 * the first simulation, but not modified before later Simulate calls.
		 */
		double I_stim_;
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
		//double PSConInit_AMPA_1;

		/**
		 * Impulse to add to DG_NMDA_1 on spike arrival to evoke unit-amplitude
		 * conductance excursion.
		 */
		//double PSConInit_NMDA_1;

		/**
		 * Impulse to add to DG_GABAA on spike arrival to evoke unit-amplitude
		 * conductance excursion.
		 */
		//double PSConInit_GABAA;

		unsigned int  RefractoryCounts_;
	};

	// Access functions for UniversalDataLogger -------------------------------

	//! Read out state vector elements, used by UniversalDataLogger
	template <State_::StateVecElems_ elem>
	double get_y_elem_()    const { return S_.y[elem]; }
	double get_I_() 				const { return S_.I_; }
	double get_I_AMPA_1_() 		const { return S_.I_AMPA_1_; 		}
	double get_I_AMPA_2_() 		const { return S_.I_AMPA_2_; 		}
	double get_I_NMDA_1_() 		const { return S_.I_NMDA_1_;    }
	double get_I_GABAA_1_() const { return S_.I_GABAA_1_; }
	double get_I_GABAA_2_() const { return S_.I_GABAA_2_; }
	double get_I_V_clamp_() const { return S_.I_V_clamp_; }

	// ---------------------------------------------------------------- 

	Parameters_ P_;
	State_      S_;
	Variables_  V_;
	Buffers_    B_;

	//! Mapping of recordables names to access functions
	static nest::RecordablesMap<my_aeif_cond_exp> recordablesMap_;
};

//inline
//nest::port mynest::my_aeif_cond_exp::check_connection(nest::Connection& c, nest::port receptor_type)
//{
//	nest::SpikeEvent e;
//	e.set_sender(*this);
//	c.check_event(e);
//	return c.get_target()->connect_sender(e, receptor_type);
//}
//
//inline
//nest::port mynest::my_aeif_cond_exp::connect_sender(nest::SpikeEvent&, nest::port receptor_type)
//{
//	// If receptor type is less than 1 =(MIN_SPIKE_RECEPTOR) or greater or equal to 4
//	// (=SUP_SPIKE_RECEPTOR) then provided receptor type is not a spike receptor.
//	if ( receptor_type < MIN_SPIKE_RECEPTOR || receptor_type >= SUP_SPIKE_RECEPTOR )
//		// Unknown receptor type is less than 0 or greater than 6
//		// (SUP_CURR_RECEPTOR).
//		if ( receptor_type < 0 || receptor_type >= SUP_CURR_RECEPTOR )
//			throw nest::UnknownReceptorType(receptor_type, get_name());
//	// Otherwise it is a current receptor or receptor 0 (data logging request
//	// not used here and therefore incompatible.
//		else
//			throw nest::IncompatibleReceptorType(receptor_type, get_name(), "SpikeEvent");
//	// If we arrive here the receptor type is a spike receptor and either 1, 2 or 3 e.i.
//	// greater or equal to MIN_SPIKE_RECEPTOR = 1, and less than SUP_SPIKE_RECEPTOR
//	// = 4. Then 0, 1, or 2 is returned.
//	return receptor_type - MIN_SPIKE_RECEPTOR;
//}
//
//inline
//nest::port mynest::my_aeif_cond_exp::connect_sender(nest::CurrentEvent&, nest::port receptor_type)
//{
//	// If receptor type is less than 4 (MIN_CURR_RECEPTOR) or greater or equal
//	// to 5 (SUP_CURR_RECEPTOR) the provided receptor type is not current
//	// receptor.
//	if ( receptor_type < MIN_CURR_RECEPTOR || receptor_type >= SUP_CURR_RECEPTOR )
//		// If receptor is not a current receptor but still a receptor type that is
//		// the receptor type is greater or equal to 0 or less than 3
//		// (MIN_CURR_RECEPTOR).
//		if ( receptor_type >= 0 && receptor_type < MIN_CURR_RECEPTOR )
//			throw nest::IncompatibleReceptorType(receptor_type, get_name(), "CurrentEvent");
//	// Otherwise unknown receptor type.
//		else
//			throw nest::UnknownReceptorType(receptor_type, get_name());
//	//MIN_CURR_RECEPTOR =4, If here receptor type equals 4  and 0 is returned.
//	return receptor_type - MIN_CURR_RECEPTOR;
//}
//
//inline
//nest::port mynest::my_aeif_cond_exp::connect_sender(nest::DataLoggingRequest& dlr,
//		nest::port receptor_type)
//{
//	// If receptor type does not equal 0 then it is not a data logging request
//	// receptor.
//	if ( receptor_type != 0 )
//		// If not a spike or current receptor that is less than 0 or greater or
//		//  equal to 4 (SUP_CURR_RECEPTOR).
//		if ( receptor_type < 0 || receptor_type >= SUP_CURR_RECEPTOR )
//			throw nest::UnknownReceptorType(receptor_type, get_name());
//	// Otherwise it is a spike or current receptor type.
//		else
//			throw nest::IncompatibleReceptorType(receptor_type, get_name(), "DataLoggingRequest");
//	// CHANGED
//	//B_.logger_.connect_logging_device(dlr, recordablesMap_);
//	//return 0;
//
//	// TO
//	return B_.logger_.connect_logging_device(dlr, recordablesMap_);
//
//}

inline
nest::port mynest::my_aeif_cond_exp::send_test_event(nest::Node& target, nest::rport receptor_type, nest::synindex, bool)
{
  nest::SpikeEvent e;
  e.set_sender(*this);
  return target.handles_test_event(e, receptor_type);
}

inline
nest::port mynest::my_aeif_cond_exp::handles_test_event(nest::SpikeEvent&, nest::rport receptor_type)
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

inline
nest::port mynest::my_aeif_cond_exp::handles_test_event(nest::CurrentEvent&, nest::rport receptor_type)
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

inline
nest::port mynest::my_aeif_cond_exp::handles_test_event(nest::DataLoggingRequest& dlr,
		nest::rport receptor_type)
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


inline
void mynest::my_aeif_cond_exp::get_status(DictionaryDatum &d) const
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
void mynest::my_aeif_cond_exp::set_status(const DictionaryDatum &d)
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
