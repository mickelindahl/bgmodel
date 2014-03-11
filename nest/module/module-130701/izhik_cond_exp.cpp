/*
 *  izhik_cond_exp.cpp
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
 *   Izhikivich simple model described in Dynamical Systems In Neuroscience
 *   2007
 *  (C) 2010 Mikael Lindahl
 *
 */


#include "izhik_cond_exp.h"

#ifdef HAVE_GSL

#include "exceptions.h"
#include "network.h"
#include "dict.h"
#include "integerdatum.h"
#include "doubledatum.h"
#include "dictutils.h"
#include "numerics.h"
//#include "analog_data_logger_impl.h"
#include "universal_data_logger_impl.h"
#include <limits>

#include <iomanip>
#include <iostream>
#include <cstdio>

using namespace nest; //added

/* ---------------------------------------------------------------- 
 * Recordables map
 * ---------------------------------------------------------------- */

nest::RecordablesMap<mynest::izhik_cond_exp> mynest::izhik_cond_exp::recordablesMap_;

namespace nest   // template specialization must be placed in namespace
{
/*
 * Override the create() method with one call to RecordablesMap::insert_()
 * for each quantity to be recorded.
 */
template <>
void RecordablesMap<mynest::izhik_cond_exp>::create()
{
	// use standard names whereever you can for consistency!
	// Recording current seems
	insert_(names::V_m,
			&mynest::izhik_cond_exp::get_y_elem_<mynest::izhik_cond_exp::State_::V_M>);
	insert_(Name("u"),
			&mynest::izhik_cond_exp::get_y_elem_<mynest::izhik_cond_exp::State_::u>);
	insert_(Name("g_AMPA_1"),
			&mynest::izhik_cond_exp::get_y_elem_<mynest::izhik_cond_exp::State_::G_AMPA_1>);
	insert_(Name("g_NMDA_1"),
			&mynest::izhik_cond_exp::get_y_elem_<mynest::izhik_cond_exp::State_::G_NMDA_1>);
	insert_(Name("g_GABAA_1"),
			&mynest::izhik_cond_exp::get_y_elem_<mynest::izhik_cond_exp::State_::G_GABAA_1>);
	insert_(Name("g_GABAA_2"),
			&mynest::izhik_cond_exp::get_y_elem_<mynest::izhik_cond_exp::State_::G_GABAA_2>);
	insert_(Name("g_GABAA_3"),
				&mynest::izhik_cond_exp::get_y_elem_<mynest::izhik_cond_exp::State_::G_GABAA_3>);

	insert_(Name("I"        ), &mynest::izhik_cond_exp::get_I_);
	insert_(Name("I_AMPA_1"   ), &mynest::izhik_cond_exp::get_I_AMPA_1_);
	insert_(Name("I_NMDA_1"   ), &mynest::izhik_cond_exp::get_I_NMDA_1_);
	insert_(Name("I_GABAA_1"), &mynest::izhik_cond_exp::get_I_GABAA_1_);
	insert_(Name("I_GABAA_2"), &mynest::izhik_cond_exp::get_I_GABAA_2_);
	insert_(Name("I_GABAA_3"), &mynest::izhik_cond_exp::get_I_GABAA_3_);
	insert_(Name("I_V_clamp"), &mynest::izhik_cond_exp::get_I_V_clamp_);

	//insert_(names::t_ref_remaining,
	//  &mynest::izhik_cond_exp::get_r_);
}
}

/* ---------------------------------------------------------------- 
 * Iteration function
 * ---------------------------------------------------------------- */

extern "C"
inline int mynest::izhik_cond_exp_dynamics(double, const double y[], double f[], void* pnode)
{ 
	// some shorthands
	//typedef mynest::izhik_cond_exp         N;
	typedef mynest::izhik_cond_exp::State_ S;

	// get access to node so we can almost work as in a member class
	assert(pnode);
	mynest::izhik_cond_exp& node =  *(reinterpret_cast<mynest::izhik_cond_exp*>(pnode));

	// easier access to membrane potential and recovery variable

	//double V = y[S::V_M];
	const nest::double_t& V = y[ S::V_M ];
	const nest::double_t& u = y[ S::u ];

	// y[] here is---and must be---the state vector supplied by the integrator,
	// not the state vector in the node, node.S_.y[].

	// The following code is verbose for the sake of clarity. We assume that a
	// good compiler will optimize the verbosity away.


	const nest::double_t dop_AMPA_1    = 1 - node.P_.beta_I_AMPA_1*node.P_.tata_dop;
	const nest::double_t dop_NMDA_1    = 1 - node.P_.beta_I_NMDA_1*node.P_.tata_dop;
	const nest::double_t dop_GABAA_1 = 1 - node.P_.beta_I_GABAA_1*node.P_.tata_dop;
	const nest::double_t dop_GABAA_2 = 1 - node.P_.beta_I_GABAA_2*node.P_.tata_dop;
	const nest::double_t dop_GABAA_3 = 1 - node.P_.beta_I_GABAA_3*node.P_.tata_dop;

	const nest::double_t I_AMPA_1 = - y[S::G_AMPA_1] * ( V - node.P_.AMPA_1_E_rev )*dop_AMPA_1;
	const nest::double_t I_NMDA_1 = - y[S::G_NMDA_1] * ( V - node.P_.NMDA_1_E_rev )*dop_NMDA_1
      						/ ( 1 + std::exp( (node.P_.NMDA_1_Vact - V)/node.P_.NMDA_1_Sact ) );
	const nest::double_t I_GABAA_1 = - y[S::G_GABAA_1] * ( V - node.P_.GABAA_1_E_rev )*dop_GABAA_1;
	const nest::double_t I_GABAA_2 = - y[S::G_GABAA_2] * ( V - node.P_.GABAA_2_E_rev )*dop_GABAA_2;
	const nest::double_t I_GABAA_3 = - y[S::G_GABAA_3] * ( V - node.P_.GABAA_3_E_rev )*dop_GABAA_3;

	// Dopamine modulation neuron
	const nest::double_t k   = node.P_.k*(   1 - node.P_.beta_k*node.P_.tata_dop);
	const nest::double_t E_L = node.P_.E_L*( 1 - node.P_.beta_E_L*node.P_.tata_dop);
	const nest::double_t V_b = node.P_.V_b*( 1 - node.P_.beta_V_b*node.P_.tata_dop);

	// Set state variable used for recording AMPA_1, NMDA_1 and GABAA current
	// contributions with dopamine modulation
	node.S_.I_AMPA_1_    = I_AMPA_1;
	node.S_.I_NMDA_1_    = I_NMDA_1;
	node.S_.I_GABAA_1_ = I_GABAA_1;
	node.S_.I_GABAA_2_ = I_GABAA_2;
	node.S_.I_GABAA_3_ = I_GABAA_3;




	// Total input current from synapses and external input
	node.S_.I_ =  I_AMPA_1 + I_NMDA_1 + I_GABAA_1 + I_GABAA_2 + I_GABAA_3 + node.B_.I_stim_ ;

	// If voltage clamp. I_ is feed back if in v clamp mode.
	if ( node.P_.V_clamp == 1 )
		node.S_.I_V_clamp_ = k*( node.P_.V_clamp_at - E_L )*( node.P_.V_clamp_at - node.P_.V_th ) - u  + node.S_.I_ + node.P_.I_e;
	else
		node.S_.I_V_clamp_ = 0;


	// Neuron dynamics
	// dV_m/dt
	f[ S::V_M ] = ( k*( V - E_L )*(  V - node.P_.V_th ) - u + node.S_.I_ + node.P_.I_e - node.S_.I_V_clamp_ )/node.P_.C_m ;

	// du/dt
	// If V is less than Vb then b=b1 and p=p1 else b=b2 and p=p2
	if ( V < node.P_.V_b )
		f[ S::u ] = node.P_.a*( node.P_.b_1*std::pow( V - V_b, node.P_.p_1 ) - u + node.P_.u_const);
	else
		f[ S::u ] = node.P_.a*( node.P_.b_2*std::pow( V - V_b, node.P_.p_2 ) - u + node.P_.u_const);


	// Synapse dynamics
	// dg_AMPA_1/dt
	f[ S::G_AMPA_1 ] 		= -y[ S::G_AMPA_1 ] / node.P_.AMPA_1_Tau_decay;

	// dg_NMDA_1/dt
	f[ S::G_NMDA_1 ] 		= -y[ S::G_NMDA_1 ] / node.P_.NMDA_1_Tau_decay;

	// dg_GABAA_1/dt
	f[ S::G_GABAA_1 ] = -y[ S::G_GABAA_1 ] / node.P_.GABAA_1_Tau_decay;

	// dg_GABAA_2/dt
	f[ S::G_GABAA_2 ] = -y[ S::G_GABAA_2 ] / node.P_.GABAA_2_Tau_decay;

	// dg_GABAA_2/dt
	f[ S::G_GABAA_3 ] = -y[ S::G_GABAA_3 ] / node.P_.GABAA_3_Tau_decay;

	return GSL_SUCCESS;
}

/* ---------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

mynest::izhik_cond_exp::Parameters_::Parameters_()
: V_clamp   (   0    ),    		 //!< 0 or 1, no/yes for voltage clamp, i.e. when 1 then membrane voltage will be clamped at V_clamp_at
  V_clamp_at(  -70   ), 			 //!< Membrane potential in mV

  V_m	 	( -50.0  ), 		   //!< Membrane potential in mV
  E_L		( -50.0  ),        //!< Membrane resting potential
  V_th		( -40.0  ),			   //!< Instantaneous activation threshold
  C_m		(  100.0 ),        //!< Capacity of the membrane in pF
  k			(  0.7   ),				 //!< Factor determining instantaneous I-V relation
  c			( -50.0  ),				 //!< Reset value membrane potential
  kc_1      (  0.0   ),        //!< Slope of voltage reset point when u < u_kc with respect to u
  kc_2 		(  0.0   ),     	 //!< Slope of voltage reset point when u > u_kc with respect to u
  u_kc		(  0.0   ),        //!< When u < u_kc, kc = kc_1 and when u > u_kc, kc = kc_2
  V_peak	(  30.0  ),        //!< spike cut of value
  I_e       (  0.0   ),        //!< pA

  u			(  0.0   ), 			 //!< Recovery variable pA
  V_b       ( -50.0  ),			   //!< Recovery variable voltage threshold mV
  a			(  0.03  ),			   //!< Time constant slow dynamics
  b_1		( -2.0   ),				 //!< Slope factor 1 slow dynamics
  b_2		( -2.0   ),				 //!< Slope factor 2 slow dynamics
  p_1		(  1.0   ),				 //!< Polynomial voltage dependency factor 1
  p_2		(  1.0   ),				 //!< Polynomial voltage dependency factor 2
  d   		(  100.0 ),        //!< Slow variable change at spike when u > du
  u_const   (  0.0   ),				 //!< Constant current recovery variable
  u_max     (  100000.0   ),     //!< maximum value that u can take
  c_max     (  0.0   ),		     //!< maximum value that c can take

  AMPA_1_E_rev     	  (  0.0   ),  	// mV
  AMPA_1_Tau_decay 	  (  3.0   ),  	// ms

  NMDA_1_E_rev     	  (  0.0   ), 	// mV
  NMDA_1_Tau_decay 	  (  100.0 ), 	// ms
  NMDA_1_Vact      	  ( -58.0  ),  	// mV
  NMDA_1_Sact           (  2.5   ),  	// mV

  GABAA_1_E_rev       (-70.0    ),  // mV
  GABAA_1_Tau_decay   (  4.0    ),  // ms

  GABAA_2_E_rev       (-70.0    ),  // mV
  GABAA_2_Tau_decay   (  4.0    ),  	// ms

  GABAA_3_E_rev       (-70.0    ),  // mV
  GABAA_3_Tau_decay   (  4.0    ),   	// ms


  tata_dop		(0.),       //!< Proportion of open dopamine receptors. Zero as init since dopamine modulation is one then

  // With these to zero model without dopamine effect is obtained
  beta_d		(0.),     //!< Dopamine effect on d
  beta_k		(0.),     //!< Dopamine effect on k
  beta_V_b		(0.),        //!< Dopamine effect on V_b
  beta_E_L		(0.),        //!< Dopamine effect on E_L

  beta_I_AMPA_1			(0.),     //!< Dopamine effect on NMDA_1 current
  beta_I_NMDA_1			(0.),     //!< Dopamine effect on NMDA_1 current
  beta_I_GABAA_1		(0.),     //!< Dopamine effect GABAA 1 current
  beta_I_GABAA_2		(0.),     //!< Dopamine effect GABAA 2 current
  beta_I_GABAA_3		(0.)     //!< Dopamine effect GABAA 3 current

{
	recordablesMap_.create();
}

mynest::izhik_cond_exp::State_::State_(const Parameters_& p)
: I_(0.0),
  I_AMPA_1_(0.0),
  I_NMDA_1_(0.0),
  I_GABAA_1_(0.0),
  I_GABAA_2_(0.0),
  I_GABAA_3_(0.0),
  I_V_clamp_(0.0)
{
	y[V_M] = p.E_L;  // initialize to resting potential
	y[u] = p.u;   // initialize to u
	for ( size_t i = 2 ; i < STATE_VEC_SIZE ; ++i )
		y[i] = 0;
}

mynest::izhik_cond_exp::State_::State_(const State_& s)
: I_(s.I_),
  I_AMPA_1_(  s.I_AMPA_1_  ),
  I_NMDA_1_(  s.I_NMDA_1_  ),
  I_GABAA_1_(s.I_GABAA_1_),
  I_GABAA_2_(s.I_GABAA_2_),
  I_GABAA_3_(s.I_GABAA_3_),
  I_V_clamp_(s.I_V_clamp_)

{
	for ( size_t i = 0 ; i < STATE_VEC_SIZE ; ++i )
		y[i] = s.y[i];
}

mynest::izhik_cond_exp::State_& mynest::izhik_cond_exp::State_::operator=(const State_& s)
{
	if ( this == &s )  // avoid assignment to self
		return *this;

	for ( size_t i = 0 ; i < STATE_VEC_SIZE ; ++i )
		y[i] = s.y[i];

	I_         = s.I_;
	I_AMPA_1_    = s.I_AMPA_1_;
	I_NMDA_1_    = s.I_NMDA_1_;
	I_GABAA_1_ = s.I_GABAA_1_;
	I_GABAA_2_ = s.I_GABAA_2_;
	I_GABAA_3_ = s.I_GABAA_3_;
	I_V_clamp_ = s.I_V_clamp_;

	//r = s.r;
	return *this;
}

mynest::izhik_cond_exp::State_::~State_()
{
}

mynest::izhik_cond_exp::Buffers_::Buffers_(izhik_cond_exp& n)
: logger_(n),
  s_(0),
  c_(0),
  e_(0)
{
	// The other member variables are left uninitialised or are
	// automatically initialised by their default constructor.
}

mynest::izhik_cond_exp::Buffers_::Buffers_(const Buffers_&, izhik_cond_exp& n)
: logger_(n),
  s_(0),
  c_(0),
  e_(0)
{
	// The other member variables are left uninitialised or are
	// automatically initialised by their default constructor.
}

/* ---------------------------------------------------------------- 
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void mynest::izhik_cond_exp::Parameters_::get(DictionaryDatum &dd) const
{
	def<nest::double_t>(dd, "V_clamp",      V_clamp);
	def<nest::double_t>(dd, "V_clamp_at",   V_clamp_at);

	def<double>(dd,names::V_m,          V_m);
	def<nest::double_t>(dd, "E_L",      E_L);
	def<nest::double_t>(dd, "V_th",      V_th);
	def<double>(dd,names::C_m,          C_m);
	def<nest::double_t>(dd, "k",        k);
	def<nest::double_t>(dd, "c",        c);
	def<nest::double_t>(dd, "kc_1",       kc_1);
	def<nest::double_t>(dd, "kc_2",       kc_2);
	def<nest::double_t>(dd, "u_kc",       u_kc);
	def<nest::double_t>(dd, "V_peak",   V_peak);
	def<double>(dd,names::I_e,          I_e);

	def<nest::double_t>(dd, "u",      	 u);
	def<nest::double_t>(dd, "V_b",       V_b);
	def<nest::double_t>(dd, "a",         a);
	def<nest::double_t>(dd, "b_1",       b_1);
	def<nest::double_t>(dd, "b_2",       b_2);
	def<nest::double_t>(dd, "p_1",       p_1);
	def<nest::double_t>(dd, "p_2",       p_2);
	def<nest::double_t>(dd, "d",         d);
	def<nest::double_t>(dd, "u_const",   u_const);
	def<nest::double_t>(dd, "u_max",     u_max);
	def<nest::double_t>(dd, "c_max",		 c_max);

	def<nest::double_t>(dd, "AMPA_1_E_rev",         AMPA_1_E_rev);
	def<nest::double_t>(dd, "AMPA_1_Tau_decay",     AMPA_1_Tau_decay);

	def<nest::double_t>(dd, "NMDA_1_E_rev",         NMDA_1_E_rev);
	def<nest::double_t>(dd, "NMDA_1_Tau_decay",     NMDA_1_Tau_decay);
	def<nest::double_t>(dd, "NMDA_1_Vact",          NMDA_1_Vact);
	def<nest::double_t>(dd, "NMDA_1_Sact",       		NMDA_1_Sact);

	def<nest::double_t>(dd, "GABAA_1_E_rev",     	GABAA_1_E_rev);
	def<nest::double_t>(dd, "GABAA_1_Tau_decay", 	GABAA_1_Tau_decay);

	def<nest::double_t>(dd, "GABAA_2_E_rev",     	GABAA_2_E_rev);
	def<nest::double_t>(dd, "GABAA_2_Tau_decay", 	GABAA_2_Tau_decay);

	def<nest::double_t>(dd, "GABAA_3_E_rev",     	GABAA_3_E_rev);
	def<nest::double_t>(dd, "GABAA_3_Tau_decay", 	GABAA_3_Tau_decay);

	def<nest::double_t>(dd, "tata_dop",		tata_dop);       //!< Proportion of open dopamine receptors. Zero as init since dopamine modulation is one then

	def<nest::double_t>(dd, "beta_d",		beta_d);     //!< Dopamine effect on d
	def<nest::double_t>(dd, "beta_k",		beta_k);     //!< Dopamine effect on k
	def<nest::double_t>(dd, "beta_V_b",		beta_V_b);        //!< Dopamine effect on V_b
	def<nest::double_t>(dd, "beta_E_L",		beta_E_L);        //!< Dopamine effect on E_L

	def<nest::double_t>(dd, "beta_I_AMPA_1",		beta_I_AMPA_1);     //!< Dopamine effect on NMDA_1 current
	def<nest::double_t>(dd, "beta_I_NMDA_1",		beta_I_NMDA_1);     //!< Dopamine effect on NMDA_1 current
	def<nest::double_t>(dd, "beta_I_GABAA_1",	beta_I_GABAA_1);     //!< Dopamine effect GABAA 1 current
	def<nest::double_t>(dd, "beta_I_GABAA_2",	beta_I_GABAA_2);     //!< Dopamine effect GABAA 2 current
	def<nest::double_t>(dd, "beta_I_GABAA_3",	beta_I_GABAA_3);     //!< Dopamine effect GABAA 3 current

}

void mynest::izhik_cond_exp::Parameters_::set(const DictionaryDatum& dd)
{
	updateValue<nest::double_t>(dd, "V_clamp",      V_clamp);
	updateValue<nest::double_t>(dd, "V_clamp_at",   V_clamp_at);

	// allow setting the membrane potential
	updateValue<double>(dd,names::V_m,          V_m);
	updateValue<nest::double_t>(dd, "E_L",      E_L);
	updateValue<nest::double_t>(dd, "V_th",      V_th);
	updateValue<double>(dd,names::C_m,          C_m);
	updateValue<nest::double_t>(dd, "c",        c);
	updateValue<nest::double_t>(dd, "k",        k);
	updateValue<nest::double_t>(dd, "kc_1",       kc_1 );
	updateValue<nest::double_t>(dd, "kc_2",       kc_2 );
	updateValue<nest::double_t>(dd, "u_kc",       u_kc );
	updateValue<nest::double_t>(dd, "V_peak",   V_peak);
	updateValue<double>(dd,names::I_e,          I_e);

	updateValue<nest::double_t>(dd, "u",         u);
	updateValue<nest::double_t>(dd, "V_b",       V_b);
	updateValue<nest::double_t>(dd, "a",         a);
	updateValue<nest::double_t>(dd, "b_1",       b_1);
	updateValue<nest::double_t>(dd, "b_2",       b_2);
	updateValue<nest::double_t>(dd, "p_1",       p_1);
	updateValue<nest::double_t>(dd, "p_2",       p_2);
	updateValue<nest::double_t>(dd, "d",         d);
	updateValue<nest::double_t>(dd, "u_const",   u_const);
	updateValue<nest::double_t>(dd, "u_max",     u_max);
	updateValue<nest::double_t>(dd, "c_max",		 c_max);

	updateValue<nest::double_t>(dd, "AMPA_1_E_rev",        AMPA_1_E_rev);
	updateValue<nest::double_t>(dd, "AMPA_1_Tau_decay",    AMPA_1_Tau_decay);

	updateValue<nest::double_t>(dd, "NMDA_1_E_rev",        NMDA_1_E_rev);
	updateValue<nest::double_t>(dd, "NMDA_1_Tau_decay",    NMDA_1_Tau_decay);
	updateValue<nest::double_t>(dd, "NMDA_1_Vact",         NMDA_1_Vact);
	updateValue<nest::double_t>(dd, "NMDA_1_Sact",         NMDA_1_Sact);

	updateValue<nest::double_t>(dd, "GABAA_1_E_rev",     GABAA_1_E_rev);
	updateValue<nest::double_t>(dd, "GABAA_1_Tau_decay", GABAA_1_Tau_decay);

	updateValue<nest::double_t>(dd, "GABAA_2_E_rev",     GABAA_2_E_rev);
	updateValue<nest::double_t>(dd, "GABAA_2_Tau_decay", GABAA_2_Tau_decay);

	updateValue<nest::double_t>(dd, "GABAA_3_E_rev",     GABAA_3_E_rev);
	updateValue<nest::double_t>(dd, "GABAA_3_Tau_decay", GABAA_3_Tau_decay);

	updateValue<nest::double_t>(dd, "tata_dop",		tata_dop);       //!< Proportion of open dopamine receptors. Zero as init since dopamine modulation is one then

	updateValue<nest::double_t>(dd, "beta_d",		beta_d);     //!< Dopamine effect on d
	updateValue<nest::double_t>(dd, "beta_k",		beta_k);     //!< Dopamine effect on k
	updateValue<nest::double_t>(dd, "beta_V_b",		beta_V_b);        //!< Dopamine effect on V_b
	updateValue<nest::double_t>(dd, "beta_E_L",		beta_E_L);        //!< Dopamine effect on E_L

	updateValue<nest::double_t>(dd, "beta_I_AMPA_1",		beta_I_AMPA_1);     //!< Dopamine effect on NMDA_1 current
	updateValue<nest::double_t>(dd, "beta_I_NMDA_1",		beta_I_NMDA_1);     //!< Dopamine effect on NMDA_1 current
	updateValue<nest::double_t>(dd, "beta_I_GABAA_1",	beta_I_GABAA_1);     //!< Dopamine effect GABAA 1 current
	updateValue<nest::double_t>(dd, "beta_I_GABAA_2",	beta_I_GABAA_2);     //!< Dopamine effect GABAA 2 current
	updateValue<nest::double_t>(dd, "beta_I_GABAA_3",	beta_I_GABAA_3);     //!< Dopamine effect GABAA 3 current

	if ( E_L > V_th )
		throw BadProperty("Reset potential must be smaller than instantaneous threshold.");

	if ( C_m <= 0 )
		throw BadProperty("Capacitance must be strictly positive.");

	//if ( t_ref < 0 )
	//  throw BadProperty("Refractory time cannot be negative.");

	if ( AMPA_1_Tau_decay    <= 0 ||
			NMDA_1_Tau_decay    <= 0 ||
			GABAA_1_Tau_decay <= 0 ||
			GABAA_2_Tau_decay <= 0 ||
			GABAA_3_Tau_decay <= 0 )
		throw BadProperty("All time constants must be strictly positive.");


}

void mynest::izhik_cond_exp::State_::get(DictionaryDatum &d) const
{
	def<double>(d, names::V_m, y[V_M]); // Membrane potential
	def<nest::double_t>(d, "u", y[u]); // Recovery variable
}

void mynest::izhik_cond_exp::State_::set(const DictionaryDatum& d, const Parameters_&)
{
	updateValue<double>(d, names::V_m, y[V_M]);
	updateValue<nest::double_t>(d, "u", y[u]); // Recovery variable
}


/* ---------------------------------------------------------------- 
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */

mynest::izhik_cond_exp::izhik_cond_exp()
: Archiving_Node(),
  P_(),
  S_(P_),
  B_(*this)
{
}

mynest::izhik_cond_exp::izhik_cond_exp(const izhik_cond_exp& n)
: Archiving_Node(n),
  P_(n.P_),
  S_(n.S_),
  B_(n.B_, *this)
{
}

mynest::izhik_cond_exp::~izhik_cond_exp()
{
	// GSL structs only allocated by init_nodes_(), so we need to protect destruction
	if ( B_.s_ ) gsl_odeiv_step_free(B_.s_);
	if ( B_.c_ ) gsl_odeiv_control_free(B_.c_);
	if ( B_.e_ ) gsl_odeiv_evolve_free(B_.e_);
}

/* ---------------------------------------------------------------- 
 * Node initialization functions
 * ---------------------------------------------------------------- */

void mynest::izhik_cond_exp::init_node_(const Node& proto)
{
	const izhik_cond_exp& pr = downcast<izhik_cond_exp>(proto);
	P_ = pr.P_;
	S_ = pr.S_;
}

void mynest::izhik_cond_exp::init_state_(const Node& proto)
{
	const izhik_cond_exp& pr = downcast<izhik_cond_exp>(proto);
	S_ = pr.S_;
}

void mynest::izhik_cond_exp::init_buffers_()
{

	B_.spikes_AMPA_1_.clear();       // includes resize
	B_.spikes_NMDA_1_.clear();       // includes resize
	B_.spikes_GABAA_1_.clear();    // includes resize
	B_.spikes_GABAA_2_.clear();    // includes resize
	B_.spikes_GABAA_3_.clear();    // includes resize
	B_.currents_.clear();          // includes resize

	B_.logger_.reset();


	nest::Archiving_Node::clear_history();

	B_.step_ = Time::get_resolution().get_ms();

	// We must integrate this model with high-precision to obtain decent results
	B_.IntegrationStep_ = std::min(0.01, B_.step_);
	//B_.IntegrationStep_ = B_.step_;

	static const gsl_odeiv_step_type* T1 = gsl_odeiv_step_rkf45;

	if ( B_.s_ == 0 )
		B_.s_ = gsl_odeiv_step_alloc (T1, State_::STATE_VEC_SIZE);
	else
		gsl_odeiv_step_reset(B_.s_);

	// Lower tolerance
	if ( B_.c_ == 0 )
		B_.c_ = gsl_odeiv_control_y_new (1e-6, 1e-6); // Changed from (1e-3, 0)
	else
		gsl_odeiv_control_init(B_.c_, 1e-6, 1e-6, 0.0, 1.0); // Changed from ( 1e-3, 0.0, 1.0, 0.0)

	/*
	if ( B_.c_ == 0 )
			B_.c_ = gsl_odeiv_control_y_new (1e-3, 0.0);
	else
			gsl_odeiv_control_init(B_.c_,  1e-3, 0.0, 1.0, 0.0);
	 */

	if ( B_.e_ == 0 )
		B_.e_ = gsl_odeiv_evolve_alloc(State_::STATE_VEC_SIZE);
	else
		gsl_odeiv_evolve_reset(B_.e_);

	B_.sys_.function  = izhik_cond_exp_dynamics;
	B_.sys_.jacobian  = NULL;
	B_.sys_.dimension = State_::STATE_VEC_SIZE;
	B_.sys_.params    = reinterpret_cast<void*>(this);

	B_.I_stim_ = 0.0;
}

// As in aeif_cond_exp.cpp but without refactory
void mynest::izhik_cond_exp::calibrate()
{
	B_.logger_.init();  // ensures initialization in case mm connected after Simulate

	//assert(V_.RefractoryCounts >= 0);  // since t_ref >= 0, this can only fail in error
}

/* ---------------------------------------------------------------- 
 * Update and spike handling functions
 * ---------------------------------------------------------------- */

void mynest::izhik_cond_exp::update(Time const & origin, const nest::long_t from, const nest::long_t to)
{

	assert(to >= 0 && (delay) from < Scheduler::get_min_delay());
	assert(from < to);


	for ( nest::long_t lag = from ; lag < to ; ++lag )
	{

		double t = 0.0;


		// numerical integration with adaptive step size control:
		// ------------------------------------------------------
		// gsl_odeiv_evolve_apply performs only a single numerical
		// integration step, starting from t and bounded by step;
		// the while-loop ensures integration over the whole simulation
		// step (0, step] if more than one integration step is needed due
		// to a small integration step size;
		// note that (t+IntegrationStep > step) leads to integration over
		// (t, step] and afterwards setting t to step, but it does not
		// enforce setting IntegrationStep to step-t; this is of advantage
		// for a consistent and efficient integration across subsequent
		// simulation intervals
		while ( t < B_.step_ )
		{
			const int status = gsl_odeiv_evolve_apply(B_.e_, B_.c_, B_.s_,
					&B_.sys_,             // system of ODE
					&t,                   // from t
					B_.step_,            // to t <= step
					&B_.IntegrationStep_, // integration step size
					S_.y); 	         // neuronal state

			if ( status != GSL_SUCCESS )
				throw GSLSolverFailure(get_name(), status);



			// spikes are handled inside the while-loop
			// due to spike-driven adaptation
			if ( S_.y[State_::V_M] >= P_.V_peak )
			{

				//S_.y[State_::V_M]=P_.V_peak;
				//B_.logger_.record_data(origin.get_steps());


				// v is reset to c + u*kc
				// S_.y[State_::V_M] = P_.c + S_.y[State_::u]*P_.kc;
				if ( S_.y[State_::u] < P_.u_kc  )
					S_.y[State_::V_M] = P_.c + S_.y[State_::u]*P_.kc_1;
				else
					S_.y[State_::V_M] = P_.c + S_.y[State_::u]*P_.kc_2;

				if ( S_.y[State_::V_M] > P_.c_max )
					S_.y[State_::V_M] = P_.c_max;

				// Update u with dopamine modulation
				S_.y[State_::u] +=P_.d*( 1 - P_.beta_d*P_.tata_dop );

				if ( S_.y[State_::u] > P_.u_max )
					S_.y[State_::u] = P_.u_max;

				// log spike with Archiving_Node
				set_spiketime(Time::step(origin.get_steps()+lag+1));
				SpikeEvent se;
				network()->send(*this, se, lag);
			}
		}

		// Here incomming spikes are added. It is setup such that AMPA_1, NMDA_1
		// and GABA recieves from one receptor each.
		S_.y[State_::G_AMPA_1]    += B_.spikes_AMPA_1_.get_value(lag);
		S_.y[State_::G_NMDA_1]    += B_.spikes_NMDA_1_.get_value(lag);
		S_.y[State_::G_GABAA_1] += B_.spikes_GABAA_1_.get_value(lag);
		S_.y[State_::G_GABAA_2] += B_.spikes_GABAA_2_.get_value(lag);
		S_.y[State_::G_GABAA_3] += B_.spikes_GABAA_3_.get_value(lag);

		// set new input current
		B_.I_stim_ = B_.currents_.get_value(lag);

		// log state data
		B_.logger_.record_data(origin.get_steps() + lag);

	}
}

void mynest::izhik_cond_exp::handle(SpikeEvent & e)
{
	assert(e.get_delay() > 0);
	// Assert that port is 0 or 1 (SUP_SPIKE_RECEPTOR (3)- MIN_SPIKE_RECEPTOR (1)
	// = 2). AMPA_1 =1, NMDA_1 =2
	assert(0 <= e.get_rport() && e.get_rport() < SUP_SPIKE_RECEPTOR - MIN_SPIKE_RECEPTOR);

	// If AMPA_1
	if (e.get_rport() == AMPA_1 - MIN_SPIKE_RECEPTOR)
		B_.spikes_AMPA_1_.add_value(e.get_rel_delivery_steps(network()->get_slice_origin()),
				e.get_weight() * e.get_multiplicity() );

	// If NMDA_1
	else if (e.get_rport() == NMDA_1 - MIN_SPIKE_RECEPTOR)
		B_.spikes_NMDA_1_.add_value(e.get_rel_delivery_steps(network()->get_slice_origin()),
				e.get_weight() * e.get_multiplicity() );

	// If GABAA_1
	else if (e.get_rport() == GABAA_1 - MIN_SPIKE_RECEPTOR)
		B_.spikes_GABAA_1_.add_value(e.get_rel_delivery_steps(network()->get_slice_origin()),
				e.get_weight() * e.get_multiplicity() );

	// If GABAA_2
	else if (e.get_rport() == GABAA_2 - MIN_SPIKE_RECEPTOR)
		B_.spikes_GABAA_2_.add_value(e.get_rel_delivery_steps(network()->get_slice_origin()),
				e.get_weight() * e.get_multiplicity() );

	// If GABAA_3
	else if (e.get_rport() == GABAA_3 - MIN_SPIKE_RECEPTOR)
		B_.spikes_GABAA_3_.add_value(e.get_rel_delivery_steps(network()->get_slice_origin()),
				e.get_weight() * e.get_multiplicity() );

}

void mynest::izhik_cond_exp::handle(CurrentEvent& e)
{
	assert(e.get_delay() > 0);

	// add weighted current; HEP 2002-10-04
	B_.currents_.add_value(e.get_rel_delivery_steps(network()->get_slice_origin()),
			e.get_weight() * e.get_current());

	assert(e.get_delay() > 0);
	// Assert that port is 0 (SUP_SPIKE_RECEPTOR (4) - MIN_SPIKE_RECEPTOR (3) = 1)
	assert(0 <= e.get_rport() && e.get_rport() < SUP_CURR_RECEPTOR - MIN_CURR_RECEPTOR);

}

void mynest::izhik_cond_exp::handle(DataLoggingRequest& e)
{
	B_.logger_.handle(e);
}

#endif //HAVE_GSL
