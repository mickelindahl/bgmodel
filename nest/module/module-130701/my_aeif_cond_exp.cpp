/*
 *  my_aeif_cond_exp.cpp
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
 */

#include "my_aeif_cond_exp.h"
//#include "nest_names.h"

#ifdef HAVE_GSL_1_11

#include "universal_data_logger_impl.h"

#include "exceptions.h"
#include "network.h"
#include "dict.h"
#include "integerdatum.h"
#include "doubledatum.h"
#include "dictutils.h"
#include "numerics.h"
#include <limits>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <cstdio>

//using namespace nest; //added
/* ---------------------------------------------------------------- 
 * Recordables map
 * ---------------------------------------------------------------- */

nest::RecordablesMap<mynest::my_aeif_cond_exp> mynest::my_aeif_cond_exp::recordablesMap_;

namespace nest
{
/*
 * template specialization must be placed in namespace
 *
 * Override the create() method with one call to RecordablesMap::insert_()
 * for each quantity to be recorded.
 */
template <>
void RecordablesMap<mynest::my_aeif_cond_exp>::create()
{
	// use standard names whereever you can for consistency!
	// Recording current seems
	insert_(names::V_m,
			&mynest::my_aeif_cond_exp::get_y_elem_<mynest::my_aeif_cond_exp::State_::V_M>);
	insert_(Name("u"),
			&mynest::my_aeif_cond_exp::get_y_elem_<mynest::my_aeif_cond_exp::State_::u>);
	insert_(Name("g_AMPA_1"),
			&mynest::my_aeif_cond_exp::get_y_elem_<mynest::my_aeif_cond_exp::State_::G_AMPA_1>);
	insert_(Name("g_AMPA_2"),
			&mynest::my_aeif_cond_exp::get_y_elem_<mynest::my_aeif_cond_exp::State_::G_AMPA_2>);
	insert_(Name("g_NMDA_1"),
			&mynest::my_aeif_cond_exp::get_y_elem_<mynest::my_aeif_cond_exp::State_::G_NMDA_1>);
	insert_(Name("g_GABAA_1"),
			&mynest::my_aeif_cond_exp::get_y_elem_<mynest::my_aeif_cond_exp::State_::G_GABAA_1>);
	insert_(Name("g_GABAA_2"),
			&mynest::my_aeif_cond_exp::get_y_elem_<mynest::my_aeif_cond_exp::State_::G_GABAA_2>);

	insert_(Name("I"        ), &mynest::my_aeif_cond_exp::get_I_);
	insert_(Name("I_AMPA_1"   ), &mynest::my_aeif_cond_exp::get_I_AMPA_1_);
	insert_(Name("I_AMPA_2"   ), &mynest::my_aeif_cond_exp::get_I_AMPA_2_);
	insert_(Name("I_NMDA_1"   ), &mynest::my_aeif_cond_exp::get_I_NMDA_1_);
	insert_(Name("I_GABAA_1"), &mynest::my_aeif_cond_exp::get_I_GABAA_1_);
	insert_(Name("I_GABAA_2"), &mynest::my_aeif_cond_exp::get_I_GABAA_2_);
	insert_(Name("I_V_clamp"), &mynest::my_aeif_cond_exp::get_I_V_clamp_);

	//insert_(names::t_ref_remaining,
	//  &mynest::my_aeif_cond_exp::get_r_);
}
}

extern "C"
int mynest::my_aeif_cond_exp_dynamics (double, const double y[], double f[], void* pnode)
{
	// a shorthand
	typedef mynest::my_aeif_cond_exp::State_ S;

	// get access to node so we can almost work as in a member function
	assert(pnode);
	mynest::my_aeif_cond_exp& node =  *(reinterpret_cast<mynest::my_aeif_cond_exp*>(pnode));

	// y[] here is---and must be---the state vector supplied by the integrator,
	// not the state vector in the node, node.S_.y[].

	// The following code is verbose for the sake of clarity. We assume that a
	// good compiler will optimize the verbosity away ...

	// shorthand for state variables
	const nest::double_t& V     = y[S::V_M  ];
	const nest::double_t& u     = y[S::u    ];

	// The following code is verbose for the sake of clarity. We assume that a
	// good compiler will optimize the verbosity away.

	const nest::double_t dop_AMPA_1    = 1 - node.P_.beta_I_AMPA_1*node.P_.tata_dop;
	const nest::double_t dop_AMPA_2    = 1 - node.P_.beta_I_AMPA_2*node.P_.tata_dop;
	const nest::double_t dop_NMDA_1    = 1 - node.P_.beta_I_NMDA_1*node.P_.tata_dop;
	const nest::double_t dop_GABAA_1 = 1 - node.P_.beta_I_GABAA_1*node.P_.tata_dop;
	const nest::double_t dop_GABAA_2 = 1 - node.P_.beta_I_GABAA_2*node.P_.tata_dop;

	const nest::double_t I_AMPA_1 = - y[S::G_AMPA_1] * ( V - node.P_.AMPA_1_E_rev )*dop_AMPA_1;
	const nest::double_t I_AMPA_2 = - y[S::G_AMPA_2] * ( V - node.P_.AMPA_2_E_rev )*dop_AMPA_2;
	const nest::double_t I_NMDA_1 = - y[S::G_NMDA_1] * ( V - node.P_.NMDA_1_E_rev )*dop_NMDA_1
      						/ ( 1 + std::exp( (node.P_.NMDA_1_Vact - V)/node.P_.NMDA_1_Sact ) );
	const nest::double_t I_GABAA_1 = - y[S::G_GABAA_1] * ( V - node.P_.GABAA_1_E_rev )*dop_GABAA_1;
	const nest::double_t I_GABAA_2 = - y[S::G_GABAA_2] * ( V - node.P_.GABAA_2_E_rev )*dop_GABAA_2;


	// Dopamine modulation neuron
	const nest::double_t E_L = node.P_.E_L*( 1 - node.P_.beta_E_L*node.P_.tata_dop);
	const nest::double_t V_a = node.P_.V_a*( 1 - node.P_.beta_V_a*node.P_.tata_dop);

		// Set state variable used for recording AMPA_1, NMDA_1 and GABAA current
	// contributions
	node.S_.I_AMPA_1_    = I_AMPA_1;
	node.S_.I_AMPA_2_    = I_AMPA_2;
	node.S_.I_NMDA_1_    = I_NMDA_1;
	node.S_.I_GABAA_1_ = I_GABAA_1;
	node.S_.I_GABAA_2_ = I_GABAA_2;

	// Total input current from synapses and external input
	node.S_.I_ = I_AMPA_1 +I_AMPA_2 + I_NMDA_1 + I_GABAA_1 + I_GABAA_2 + node.B_.I_stim_ ;

	const nest::double_t I_spike   = node.P_.Delta_T * std::exp((V - node.P_.V_th) / node.P_.Delta_T);

	// dv/dt
	f[S::V_M  ] = ( -node.P_.g_L *( (V-E_L) - I_spike)
		            - u + node.P_.I_e + node.S_.I_) / node.P_.C_m;


	// If V is less than V_a then a=a_1 and else a=a_2
	//double a; // Short cut
	if ( V < node.P_.V_a )
		//a=node.P_.a_1;
		f[S::u    ] = ( node.P_.a_1 * (V - V_a) - u ) / node.P_.tau_w;
	else
		//a=node.P_.a_2;
		f[S::u    ] = ( node.P_.a_2 * (V - V_a) - u ) / node.P_.tau_w;

	// Adaptation current u.
	//f[S::u    ] = ( a * (V - node.P_.V_a) - u ) / node.P_.tau_w;

	// Synapse dynamics
	// dg_AMPA_1/dt
	f[ S::G_AMPA_1 ]  = -y[ S::G_AMPA_1 ] / node.P_.AMPA_1_Tau_decay;

	// Synapse dynamics
	// dg_AMPA_2/dt
	f[ S::G_AMPA_2 ]  = -y[ S::G_AMPA_2 ] / node.P_.AMPA_2_Tau_decay;

	// dg_NMDA_1/dt
	f[ S::G_NMDA_1 ]  = -y[ S::G_NMDA_1 ] / node.P_.NMDA_1_Tau_decay;

	// dg_GABAA_1/dt
	f[ S::G_GABAA_1 ] = -y[ S::G_GABAA_1 ] / node.P_.GABAA_1_Tau_decay;

	// dg_GABAA_2/dt
	f[ S::G_GABAA_2 ] = -y[ S::G_GABAA_2 ] / node.P_.GABAA_2_Tau_decay;

	return GSL_SUCCESS;
}

/* ---------------------------------------------------------------- 
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

mynest::my_aeif_cond_exp::Parameters_::Parameters_()
: V_peak_    (   0.0 ), // mV
  V_reset_   ( -60.0 ), // mV
  t_ref_     (   0.0 ), // ms
  g_L        (  30.0 ), // nS
  C_m        ( 281.0 ), // pF
  E_L        ( -70.6 ), // mV
  Delta_T    (   2.0 ), // mV
  tau_w      ( 144.0 ), // ms

  //a          (   4.0 ), // nS
  a_1          (   4.0 ), // nS
  a_2          (   4.0 ), // nS
  V_a          (  -70.6 ), // nS

  b          (  80.5 ), // pA
  V_th       ( -50.4 ), // mV
  I_e        (   0.0 ), // pA
  gsl_error_tol( 1e-6),
  

	V_reset_slope1          (0.0), // Slope of v rested point
	V_reset_slope2          (0.0), // Slope of v rested point
	V_reset_max_slope1   (0.0), // mV Max increase of v reset point
	V_reset_max_slope2   (0.0), // mV Max increase of v reset point

  AMPA_1_E_rev     			(  0.0   ),  	// mV
  AMPA_1_Tau_decay 			(  3.0   ),  	// ms

  AMPA_2_E_rev     			(  0.0   ),  	// mV
  AMPA_2_Tau_decay 			(  3.0   ),  	// ms

  NMDA_1_E_rev     			(  0.0   ), 	// mV
  NMDA_1_Tau_decay 			(  100.0 ), 	// ms
  NMDA_1_Vact      			( -58.0  ),  	// mV
  NMDA_1_Sact           (  2.5   ),  	// mV

  GABAA_1_E_rev       (-70.0    ),  // mV
  GABAA_1_Tau_decay   (  4.0    ),  // ms

  GABAA_2_E_rev       (-70.0    ),  // mV
  GABAA_2_Tau_decay   (  4.0    ),  	// ms

  tata_dop		(0.),       //!< Proportion of open dopamine receptors. Zero as init since dopamine modulation is one then

  // With these to zero model without dopamine effect is obtained
  beta_V_a		(0.),        //!< Dopamine effect on V_b
  beta_E_L		(0.),        //!< Dopamine effect on E_L



  beta_I_AMPA_1			(0.),     //!< Dopamine effect on NMDA_1 current
  beta_I_AMPA_2			(0.),     //!< Dopamine effect on NMDA_1 current
  beta_I_NMDA_1			(0.),     //!< Dopamine effect on NMDA_1 current
  beta_I_GABAA_1		(0.),     //!< Dopamine effect GABAA 1 current
  beta_I_GABAA_2		(0.)     //!< Dopamine effect GABAA 2 current
{
	recordablesMap_.create();
}

mynest::my_aeif_cond_exp::State_::State_(const Parameters_ &p)
: I_(0.0),
  I_AMPA_1_(0.0),
  I_AMPA_2_(0.0),
  I_NMDA_1_(0.0),
  I_GABAA_1_(0.0),
  I_GABAA_2_(0.0),
  I_V_clamp_(0.0), 
  r_(0)

{
	y[0] = p.E_L;   // initialize to resting potential
	for ( size_t i = 1; i <STATE_VEC_SIZE; ++i )
		y[i] = 0;
}

mynest::my_aeif_cond_exp::State_::State_(const State_ &s)
:I_(s.I_),
  I_AMPA_1_(  s.I_AMPA_1_  ),
  I_AMPA_2_(  s.I_AMPA_2_  ),
  I_NMDA_1_(  s.I_NMDA_1_  ),
  I_GABAA_1_(s.I_GABAA_1_),
  I_GABAA_2_(s.I_GABAA_2_),
  I_V_clamp_(s.I_V_clamp_), 
r_(s.r_)

{
	for ( size_t i = 0; i < STATE_VEC_SIZE; ++i )
		y[i] = s.y[i];
}

mynest::my_aeif_cond_exp::State_& mynest::my_aeif_cond_exp::State_::operator=(const State_ &s)
{
	assert(this != &s);  // would be bad logical error in program

	for ( size_t i = 0; i < STATE_VEC_SIZE; ++i )
		y[i] = s.y[i];
		
			I_         = s.I_;
	I_AMPA_1_    = s.I_AMPA_1_;
	I_AMPA_2_    = s.I_AMPA_2_;
	I_NMDA_1_    = s.I_NMDA_1_;
	I_GABAA_1_ = s.I_GABAA_1_;
	I_GABAA_2_ = s.I_GABAA_2_;
	I_V_clamp_ = s.I_V_clamp_;
	r_ = s.r_;
	return *this;
}
mynest::my_aeif_cond_exp::State_::~State_()
{
}

mynest::my_aeif_cond_exp::Buffers_::Buffers_(my_aeif_cond_exp &n)
: logger_(n),
  s_(0),
  c_(0),
  e_(0)
{
	// Initialization of the remaining members is deferred to
	// init_buffers_().
}

mynest::my_aeif_cond_exp::Buffers_::Buffers_(const Buffers_ &, my_aeif_cond_exp &n)
: logger_(n),
  s_(0),
  c_(0),
  e_(0)
{
	// Initialization of the remaining members is deferred to
	// init_buffers_().
}

/* ---------------------------------------------------------------- 
 * Paramater and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void mynest::my_aeif_cond_exp::Parameters_::get(DictionaryDatum &dd) const
{
	def<double>(dd,nest::names::C_m,        C_m);
	def<double>(dd,nest::names::V_th,       V_th);
	def<double>(dd,nest::names::t_ref,      t_ref_);
	def<double>(dd,nest::names::g_L,        g_L);
	def<double>(dd,nest::names::E_L,        E_L);
	def<double>(dd,nest::names::V_reset,    V_reset_);
	//def<double>(dd,names::a,          a);
	def<nest::double_t>(dd, "a_1",       a_1);
	def<nest::double_t>(dd, "a_2",       a_2);
	def<nest::double_t>(dd, "V_a",       V_a);
	def<double>(dd,nest::names::b,          b);
	def<double>(dd,nest::names::Delta_T,    Delta_T);
	def<double>(dd,nest::names::tau_w,      tau_w);
	def<double>(dd,nest::names::I_e,        I_e);
	def<double>(dd,nest::names::V_peak,     V_peak_);
	def<double>(dd,nest::names::gsl_error_tol, gsl_error_tol);
	
	
	def<nest::double_t>(dd, "V_reset_slope1",       			  V_reset_slope1);
	def<nest::double_t>(dd, "V_reset_slope2",       			  V_reset_slope2);
    def<nest::double_t>(dd, "V_reset_max_slope1",       V_reset_max_slope1);
    def<nest::double_t>(dd, "V_reset_max_slope2",       V_reset_max_slope2);

	def<nest::double_t>(dd, "AMPA_1_E_rev",         AMPA_1_E_rev);
	def<nest::double_t>(dd, "AMPA_1_Tau_decay",     AMPA_1_Tau_decay);

	def<nest::double_t>(dd, "AMPA_2_E_rev",         AMPA_2_E_rev);
	def<nest::double_t>(dd, "AMPA_2_Tau_decay",     AMPA_2_Tau_decay);

	def<nest::double_t>(dd, "NMDA_1_E_rev",         NMDA_1_E_rev);
	def<nest::double_t>(dd, "NMDA_1_Tau_decay",     NMDA_1_Tau_decay);
	def<nest::double_t>(dd, "NMDA_1_Vact",          NMDA_1_Vact);
	def<nest::double_t>(dd, "NMDA_1_Sact",       		NMDA_1_Sact);

	def<nest::double_t>(dd, "GABAA_1_E_rev",     	GABAA_1_E_rev);
	def<nest::double_t>(dd, "GABAA_1_Tau_decay", 	GABAA_1_Tau_decay);

	def<nest::double_t>(dd, "GABAA_2_E_rev",     	GABAA_2_E_rev);
	def<nest::double_t>(dd, "GABAA_2_Tau_decay", 	GABAA_2_Tau_decay);

	def<nest::double_t>(dd, "tata_dop",		tata_dop);       //!< Proportion of open dopamine receptors. Zero as init since dopamine modulation is one then

	def<nest::double_t>(dd, "beta_V_a",		beta_V_a);        //!< Dopamine effect on V_b
	def<nest::double_t>(dd, "beta_E_L",		beta_E_L);        //!< Dopamine effect on E_L

	def<nest::double_t>(dd, "beta_I_AMPA_1",		beta_I_AMPA_1);     //!< Dopamine effect on NMDA_1 current
	def<nest::double_t>(dd, "beta_I_AMPA_2",		beta_I_AMPA_2);     //!< Dopamine effect on NMDA_1 current
	def<nest::double_t>(dd, "beta_I_NMDA_1",		beta_I_NMDA_1);     //!< Dopamine effect on NMDA_1 current
	def<nest::double_t>(dd, "beta_I_GABAA_1",	beta_I_GABAA_1);     //!< Dopamine effect GABAA 1 current
	def<nest::double_t>(dd, "beta_I_GABAA_2",	beta_I_GABAA_2);     //!< Dopamine effect GABAA 2 current
}

void mynest::my_aeif_cond_exp::Parameters_::set(const DictionaryDatum &dd)
{
	updateValue<double>(dd,nest::names::V_th,    V_th);
	updateValue<double>(dd,nest::names::V_peak,  V_peak_);
	updateValue<double>(dd,nest::names::t_ref,   t_ref_);
	updateValue<double>(dd,nest::names::E_L,     E_L);
	updateValue<double>(dd,nest::names::V_reset, V_reset_);

	updateValue<double>(dd,nest::names::C_m, C_m);
	updateValue<double>(dd,nest::names::g_L, g_L);

	//updateValue<double>(dd,names::a,       a);
	updateValue<nest::double_t>(dd, "a_1",       a_1);
	updateValue<nest::double_t>(dd, "a_2",       a_2);
	updateValue<nest::double_t>(dd, "V_a",       V_a);
	updateValue<double>(dd,nest::names::b,       b);
	updateValue<double>(dd,nest::names::Delta_T, Delta_T);
	updateValue<double>(dd,nest::names::tau_w,   tau_w);
	updateValue<double>(dd,nest::names::I_e, I_e);
	updateValue<double>(dd,nest::names::gsl_error_tol, gsl_error_tol);


	updateValue<nest::double_t>(dd, "V_reset_slope1",       V_reset_slope1);
	updateValue<nest::double_t>(dd, "V_reset_slope2",       V_reset_slope2);
	updateValue<nest::double_t>(dd, "V_reset_max_slope1",   V_reset_max_slope1);
	updateValue<nest::double_t>(dd, "V_reset_max_slope2",   V_reset_max_slope2);

    updateValue<nest::double_t>(dd, "AMPA_1_E_rev",        AMPA_1_E_rev);
	updateValue<nest::double_t>(dd, "AMPA_1_Tau_decay",    AMPA_1_Tau_decay);
    updateValue<nest::double_t>(dd, "AMPA_2_E_rev",        AMPA_2_E_rev);
	updateValue<nest::double_t>(dd, "AMPA_2_Tau_decay",    AMPA_2_Tau_decay);

	updateValue<nest::double_t>(dd, "NMDA_1_E_rev",        NMDA_1_E_rev);
	updateValue<nest::double_t>(dd, "NMDA_1_Tau_decay",    NMDA_1_Tau_decay);
	updateValue<nest::double_t>(dd, "NMDA_1_Vact",         NMDA_1_Vact);
	updateValue<nest::double_t>(dd, "NMDA_1_Sact",         NMDA_1_Sact);

	updateValue<nest::double_t>(dd, "GABAA_1_E_rev",     GABAA_1_E_rev);
	updateValue<nest::double_t>(dd, "GABAA_1_Tau_decay", GABAA_1_Tau_decay);

	updateValue<nest::double_t>(dd, "GABAA_2_E_rev",     GABAA_2_E_rev);
	updateValue<nest::double_t>(dd, "GABAA_2_Tau_decay", GABAA_2_Tau_decay);
	
	updateValue<nest::double_t>(dd, "tata_dop",		tata_dop);       //!< Proportion of open dopamine receptors. Zero as init since dopamine modulation is one then

	updateValue<nest::double_t>(dd, "beta_V_a",		beta_V_a);        //!< Dopamine effect on V_b
	updateValue<nest::double_t>(dd, "beta_E_L",		beta_E_L);        //!< Dopamine effect on E_L

	updateValue<nest::double_t>(dd, "beta_I_AMPA_1",		beta_I_AMPA_1);     //!< Dopamine effect on NMDA_1 current
	updateValue<nest::double_t>(dd, "beta_I_AMPA_2",		beta_I_AMPA_2);     //!< Dopamine effect on NMDA_1 current
	updateValue<nest::double_t>(dd, "beta_I_NMDA_1",		beta_I_NMDA_1);     //!< Dopamine effect on NMDA_1 current
	updateValue<nest::double_t>(dd, "beta_I_GABAA_1",	beta_I_GABAA_1);     //!< Dopamine effect GABAA 1 current
	updateValue<nest::double_t>(dd, "beta_I_GABAA_2",	beta_I_GABAA_2);     //!< Dopamine effect GABAA 2 current

	if ( V_reset_ >= V_peak_ )
		throw nest::BadProperty("Reset potential must be smaller than spike cut-off threshold.");

	if ( V_peak_ <= V_th )
		throw nest::BadProperty("V_peak must be larger than threshold.");

	if ( C_m <= 0 )
		throw nest::BadProperty("Capacitance must be strictly positive.");

	if ( t_ref_ < 0 )
		throw nest::BadProperty("Refractory time cannot be negative.");

	if ( AMPA_1_Tau_decay    <= 0 ||
			AMPA_2_Tau_decay    <= 0 ||
			NMDA_1_Tau_decay    <= 0 ||
			GABAA_1_Tau_decay <= 0 ||
			GABAA_2_Tau_decay <= 0 )
		throw nest::BadProperty("All time constants must be strictly positive.");

	if ( gsl_error_tol <= 0. )
		throw nest::BadProperty("The gsl_error_tol must be strictly positive.");
}

void mynest::my_aeif_cond_exp::State_::get(DictionaryDatum &d) const
{
	def<double>(d,nest::names::V_m,  y[V_M]);
	def<nest::double_t>(d, "u", y[u]); // Recovery variable
}

void mynest::my_aeif_cond_exp::State_::set(const DictionaryDatum &d, const Parameters_ &)
{
	updateValue<double>(d,nest::names::V_m,  y[V_M]);
	updateValue<nest::double_t>(d, "u", y[u]); // Recovery variable

}



/* ---------------------------------------------------------------- 
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */

mynest::my_aeif_cond_exp::my_aeif_cond_exp()
: Archiving_Node(),
  P_(),
  S_(P_),
  B_(*this)
{
	recordablesMap_.create();
}

mynest::my_aeif_cond_exp::my_aeif_cond_exp(const my_aeif_cond_exp &n)
: Archiving_Node(n),
  P_(n.P_),
  S_(n.S_),
  B_(n.B_, *this)
{
}

mynest::my_aeif_cond_exp::~my_aeif_cond_exp()
{
	// GSL structs only allocated by init_nodes_(), so we need to protect destruction
	if ( B_.s_ ) gsl_odeiv_step_free(B_.s_);
	if ( B_.c_ ) gsl_odeiv_control_free(B_.c_);
	if ( B_.e_ ) gsl_odeiv_evolve_free(B_.e_);
}

/* ---------------------------------------------------------------- 
 * Node initialization functions
 * ---------------------------------------------------------------- */

void mynest::my_aeif_cond_exp::init_node_(const Node &proto)
{
	const my_aeif_cond_exp &pr = downcast<my_aeif_cond_exp>(proto);
	P_ = pr.P_;
	S_ = pr.S_;
}

void mynest::my_aeif_cond_exp::init_state_(const Node &proto)
{
	const my_aeif_cond_exp &pr = downcast<my_aeif_cond_exp>(proto);
	S_ = pr.S_;
}

void mynest::my_aeif_cond_exp::init_buffers_()
{
	B_.spikes_AMPA_1_.clear();       // includes resize
	B_.spikes_AMPA_2_.clear();       // includes resize
	B_.spikes_NMDA_1_.clear();       // includes resize
	B_.spikes_GABAA_1_.clear();    // includes resize
	B_.spikes_GABAA_2_.clear();    // includes resize
	B_.currents_.clear();          // includes resize

	B_.logger_.reset();
	
	nest::Archiving_Node::clear_history();

	B_.step_ = nest::Time::get_resolution().get_ms();

	// We must integrate this model with high-precision to obtain decent results
	B_.IntegrationStep_ = std::min(0.01, B_.step_);

	static const gsl_odeiv_step_type* T1 = gsl_odeiv_step_rkf45;

	if ( B_.s_ == 0 )
		B_.s_ = gsl_odeiv_step_alloc (T1, State_::STATE_VEC_SIZE);
	else
		gsl_odeiv_step_reset(B_.s_);

	if ( B_.c_ == 0 )
		B_.c_ = gsl_odeiv_control_yp_new (P_.gsl_error_tol,P_.gsl_error_tol);
	else
		gsl_odeiv_control_init(B_.c_, P_.gsl_error_tol, P_.gsl_error_tol, 0.0, 1.0);

	if ( B_.e_ == 0 )
		B_.e_ = gsl_odeiv_evolve_alloc(State_::STATE_VEC_SIZE);
	else
		gsl_odeiv_evolve_reset(B_.e_);

	B_.sys_.function  = my_aeif_cond_exp_dynamics;
	B_.sys_.jacobian  = NULL;
	B_.sys_.dimension = State_::STATE_VEC_SIZE;
	B_.sys_.params    = reinterpret_cast<void*>(this);

	B_.I_stim_ = 0.0;
}

void mynest::my_aeif_cond_exp::calibrate()
{
	B_.logger_.init();  // ensures initialization in case mm connected after Simulate
	//nest::double_t t=nest::Time::ms(P_.t_ref_);
	V_.RefractoryCounts_ = 2; //nest::Time(t).get_steps();
	assert(V_.RefractoryCounts_ >= 0);  // since t_ref_ >= 0, this can only fail in error
}

/* ---------------------------------------------------------------- 
 * Update and spike handling functions
 * ---------------------------------------------------------------- */

void mynest::my_aeif_cond_exp::update(const nest::Time &origin, const nest::long_t from, const nest::long_t to)
{
	assert ( to >= 0 && (nest::delay) from < nest::Scheduler::get_min_delay() );
	assert ( from < to );
	assert ( State_::V_M == 0 );

	for ( nest::long_t lag = from; lag < to; ++lag )
	{
		double t = 0.0;

		if ( S_.r_ > 0 )
			--S_.r_;

		// numerical integration with adaptive step size control:
		// ------------------------------------------------------
		// gsl_odeiv_evolve_apply performs only a single numerical
		// integration step, starting from t and bounded by step;
		// the while-loop ensures integration over the whole simulation
		// step (0, step] if more than one integration step is needed due
		// to a small integration step size;
		// note that (t+IntegrationStep > step) leads to integration over
		// (t, step] and afterwards setting t to step, but it does not
		// enforce setting IntegrationStep to step-t
		while ( t < B_.step_ )
		{
			const int status = gsl_odeiv_evolve_apply(B_.e_, B_.c_, B_.s_,
					&B_.sys_,             // system of ODE
					&t,                   // from t
					B_.step_,             // to t <= step
					&B_.IntegrationStep_, // integration step size
					S_.y);               // neuronal state

			if ( status != GSL_SUCCESS )
				throw nest::GSLSolverFailure(get_name(), status);

			// check for unreasonable values; we allow V_M to explode
			if ( S_.y[State_::V_M] < -1e3 ||
					S_.y[State_::u  ] <    -1e6 || S_.y[State_::u] > 1e6    )
				throw nest::NumericalInstability(get_name());

			// spikes are handled inside the while-loop
			// due to spike-driven adaptation
			if ( S_.r_ > 0 )
				S_.y[State_::V_M] = P_.V_reset_;

			/*
			If V > V_peak
				if ( u < 0  )
					V = min(V_reset + u*V_reset_slope1, V_reset_max_slope1);
				else if ( u >= 0  )
					V = min(V_reset + u*V_reset_slope2, V_reset_max_slope1);
			*/
			else if ( S_.y[State_::V_M] >= P_.V_peak_ )
			{

				// Spike reset voltage point adapation
				if ( S_.y[State_::u] < 0 )
					S_.y[State_::V_M] =std::min<nest::double_t>(P_.V_reset_+ S_.y[State_::u]*P_.V_reset_slope1, P_.V_reset_max_slope1);

				else
					S_.y[State_::V_M] =std::min<nest::double_t>(P_.V_reset_+ S_.y[State_::u]*P_.V_reset_slope2, P_.V_reset_max_slope2);



				S_.y[State_::u]   += P_.b; // spike-driven adaptation
				S_.r_               = V_.RefractoryCounts_;

				set_spiketime(nest::Time::step(origin.get_steps() + lag + 1));
				nest::SpikeEvent se;
				network()->send(*this, se, lag);
			}
		}
		// Here incomming spikes are added. It is setup such that AMPA_1, NMDA_1
		// and GABA recieves from one receptor each.
		S_.y[State_::G_AMPA_1]    += B_.spikes_AMPA_1_.get_value(lag);
		S_.y[State_::G_AMPA_2]    += B_.spikes_AMPA_2_.get_value(lag);
		S_.y[State_::G_NMDA_1]    += B_.spikes_NMDA_1_.get_value(lag);
		S_.y[State_::G_GABAA_1] += B_.spikes_GABAA_1_.get_value(lag);
		S_.y[State_::G_GABAA_2] += B_.spikes_GABAA_2_.get_value(lag);


		// set new input current
		B_.I_stim_ = B_.currents_.get_value(lag);

		// log state data
		B_.logger_.record_data(origin.get_steps() + lag);
	}
}

void mynest::my_aeif_cond_exp::handle(nest::SpikeEvent & e)
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

	// If AMPA_2
	else if(e.get_rport() == AMPA_2 - MIN_SPIKE_RECEPTOR)
		B_.spikes_AMPA_2_.add_value(e.get_rel_delivery_steps(network()->get_slice_origin()),
				e.get_weight() * e.get_multiplicity() );


}

void mynest::my_aeif_cond_exp::handle(nest::CurrentEvent &e)
{
	assert ( e.get_delay() > 0 );


	// add weighted current; HEP 2002-10-04
	B_.currents_.add_value(e.get_rel_delivery_steps(network()->get_slice_origin()),
			e.get_weight() * e.get_current());

	// Assert that port is 0 (SUP_SPIKE_RECEPTOR (4) - MIN_SPIKE_RECEPTOR (3) = 1)
	assert(0 <= e.get_rport() && e.get_rport() < SUP_CURR_RECEPTOR - MIN_CURR_RECEPTOR);
}

void mynest::my_aeif_cond_exp::handle(nest::DataLoggingRequest &e)
{
	B_.logger_.handle(e);
}

#endif // HAVE_GSL_1_11
