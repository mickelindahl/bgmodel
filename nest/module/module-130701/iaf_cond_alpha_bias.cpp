/*
 *  iaf_cond_alpha_bias.cpp
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
 */


#include "iaf_cond_alpha_bias.h"

#ifdef HAVE_GSL

#include "exceptions.h"
#include "network.h"
#include "dict.h"
#include "integerdatum.h"
#include "doubledatum.h"
#include "dictutils.h"
#include "numerics.h"
#include "universal_data_logger_impl.h"
#include <limits>

#include <iomanip>
#include <iostream>
#include <cstdio>

#include <iostream>
using namespace std;

/* ---------------------------------------------------------------- 
 * Recordables map
 * ---------------------------------------------------------------- */

nest::RecordablesMap<mynest::iaf_cond_alpha_bias> mynest::iaf_cond_alpha_bias::recordablesMap_;

namespace nest   // template specialization must be placed in namespace
{
  /*
   * Override the create() method with one call to RecordablesMap::insert_() 
   * for each quantity to be recorded.
   */
  template <>
  void RecordablesMap<mynest::iaf_cond_alpha_bias>::create()
  {
    // use standard names whereever you can for consistency!
    insert_(names::V_m, 
	    &mynest::iaf_cond_alpha_bias::get_y_elem_<mynest::iaf_cond_alpha_bias::State_::V_M>);
    insert_(names::g_ex, 
	    &mynest::iaf_cond_alpha_bias::get_y_elem_<mynest::iaf_cond_alpha_bias::State_::G_EXC>);
    insert_(names::g_in, 
	    &mynest::iaf_cond_alpha_bias::get_y_elem_<mynest::iaf_cond_alpha_bias::State_::G_INH>);
    insert_(Name("z_j"), 
	    &mynest::iaf_cond_alpha_bias::get_y_elem_<mynest::iaf_cond_alpha_bias::State_::Z_J>);
    insert_(Name("e_j"), 
	    &mynest::iaf_cond_alpha_bias::get_y_elem_<mynest::iaf_cond_alpha_bias::State_::E_J>);
    insert_(Name("p_j"), 
	    &mynest::iaf_cond_alpha_bias::get_y_elem_<mynest::iaf_cond_alpha_bias::State_::P_J>);

    insert_(names::t_ref_remaining, 
	    &mynest::iaf_cond_alpha_bias::get_r_);

    insert_(Name("bias"), 
	    &mynest::iaf_cond_alpha_bias::get_bias_);

    insert_(Name("epsilon"), 
	    &mynest::iaf_cond_alpha_bias::get_epsilon_);

    insert_(Name("kappa"), 
	    &mynest::iaf_cond_alpha_bias::get_kappa_);
  }
}

/* ---------------------------------------------------------------- 
 * Iteration function
 * ---------------------------------------------------------------- */

extern "C"
inline int mynest::iaf_cond_alpha_bias_dynamics(double, const double y[], double f[], void* pnode)
{ 
  // a shorthand
  typedef mynest::iaf_cond_alpha_bias::State_ S;

  // get access to node so we can almost work as in a member function
  assert(pnode);
  const mynest::iaf_cond_alpha_bias& node =  *(reinterpret_cast<mynest::iaf_cond_alpha_bias*>(pnode));

  // y[] here is---and must be---the state vector supplied by the integrator,
  // not the state vector in the node, node.S_.y[]. 
  
  // The following code is verbose for the sake of clarity. We assume that a
  // good compiler will optimize the verbosity away ...
  const nest::double_t I_syn_exc = y[S::G_EXC] * ( y[S::V_M] - node.P_.E_ex );
  const nest::double_t I_syn_inh = y[S::G_INH] * ( y[S::V_M] - node.P_.E_in );
  const nest::double_t I_leak    = node.P_.g_L * ( y[S::V_M] - node.P_.E_L  );
  const nest::double_t I_bias    = node.P_.gain * std::log(y[S::P_J]);
  
  // dV_m/dt
  f[0] = (-I_leak - I_syn_exc - I_syn_inh + node.B_.I_stim_ + node.P_.I_e + I_bias) / node.P_.C_m;

  // d dg_exc/dt, dg_exc/dt
  f[1] = -y[S::DG_EXC] / node.P_.tau_synE;
  f[2] =  y[S::DG_EXC] - (y[S::G_EXC]/node.P_.tau_synE); 

  // d dg_exc/dt, dg_exc/dt
  f[3] = -y[S::DG_INH] / node.P_.tau_synI;
  f[4] =  y[S::DG_INH] - (y[S::G_INH]/node.P_.tau_synI); 

  f[5] = (- y[S::Z_J] + node.P_.epsilon) / node.P_.tau_j;
  f[6] = (y[S::Z_J] - y[S::E_J]) / node.P_.tau_e;
  f[7] = node.P_.kappa * (y[S::E_J] - y[S::P_J]) / node.P_.tau_p;

  return GSL_SUCCESS;
 }

/* ---------------------------------------------------------------- 
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */
    
mynest::iaf_cond_alpha_bias::Parameters_::Parameters_()
  : V_th    (-55.0    ),  // mV
    V_reset (-60.0    ),  // mV
    t_ref   (  2.0    ),  // ms
    g_L     ( 16.6667 ),  // nS
    C_m     (250.0    ),  // pF
    E_ex    (  0.0    ),  // mV
    E_in    (-85.0    ),  // mV
    E_L     (-70.0    ),  // mV
    tau_synE(  0.2    ),  // ms
    tau_synI(  2.0    ),  // ms
    I_e     (  0.0    ),  // pA
    tau_j   ( 10.0    ),  // ms
    tau_e   (100.0    ),  // ms
    tau_p   (1000.0   ),  // ms
    kappa   (1.0      ), // dopamine
    fmax    (20.0     ), 
    gain    (1.0     ), 
    bias    (0.0      ),
    epsilon (0.001     ) 
{
  recordablesMap_.create();
}

mynest::iaf_cond_alpha_bias::State_::State_(const Parameters_& p)
  : r(0), bias(0)
{
  y[V_M] = p.E_L;  // initialize to reversal potential
  for ( size_t i = 1 ; i < 5 ; ++i )
    y[i] = 0;
  y[Z_J] = 0.01;
  y[E_J] = 0.01;
  y[P_J] = 0.01;
  //y[BIAS] = 0.0;
}

mynest::iaf_cond_alpha_bias::State_::State_(const State_& s)
  : r(s.r),bias(s.bias)
{
  for ( size_t i = 0 ; i < STATE_VEC_SIZE ; ++i )
    y[i] = s.y[i];
}

mynest::iaf_cond_alpha_bias::State_& mynest::iaf_cond_alpha_bias::State_::operator=(const State_& s)
{
  if ( this == &s )  // avoid assignment to self
    return *this;

  for ( size_t i = 0 ; i < STATE_VEC_SIZE ; ++i )
    y[i] = s.y[i];

  r = s.r;
  return *this;
}

mynest::iaf_cond_alpha_bias::Buffers_::Buffers_(iaf_cond_alpha_bias& n)
  : logger_(n),
    s_(0),
    c_(0),
    e_(0)
{
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

mynest::iaf_cond_alpha_bias::Buffers_::Buffers_(const Buffers_&, iaf_cond_alpha_bias& n)
  : logger_(n),
    s_(0),
    c_(0),
    e_(0)
{
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

/* ---------------------------------------------------------------- 
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void mynest::iaf_cond_alpha_bias::Parameters_::get(DictionaryDatum &dd) const
{
  def<double>(dd,nest::names::V_th,         V_th);
  def<double>(dd,nest::names::V_reset,      V_reset);
  def<double>(dd,nest::names::t_ref,        t_ref);
  def<double>(dd,nest::names::g_L,          g_L);
  def<double>(dd,nest::names::E_L,          E_L); 
  def<double>(dd,nest::names::E_ex,         E_ex);
  def<double>(dd,nest::names::E_in,         E_in);
  def<double>(dd,nest::names::C_m,          C_m);
  def<double>(dd,nest::names::tau_syn_ex,   tau_synE);
  def<double>(dd,nest::names::tau_syn_in,   tau_synI);
  def<double>(dd,nest::names::I_e,          I_e);
  def<nest::double_t>(dd, "tau_j",      tau_j);
  def<nest::double_t>(dd, "tau_e",      tau_e);
  def<nest::double_t>(dd, "tau_p",      tau_p);
  def<nest::double_t>(dd, "kappa",      kappa);
  def<nest::double_t>(dd, "bias",      bias);
  def<nest::double_t>(dd, "gain",      gain);
  def<nest::double_t>(dd, "fmax",      fmax);
  def<nest::double_t>(dd, "epsilon",      epsilon);
}

void mynest::iaf_cond_alpha_bias::Parameters_::set(const DictionaryDatum& dd)
{
  // allow setting the membrane potential
  updateValue<double>(dd,nest::names::V_th,    V_th);
  updateValue<double>(dd,nest::names::V_reset, V_reset);
  updateValue<double>(dd,nest::names::t_ref,   t_ref);
  updateValue<double>(dd,nest::names::E_L,     E_L);
  
  updateValue<double>(dd,nest::names::E_ex,    E_ex);
  updateValue<double>(dd,nest::names::E_in,    E_in);
  
  updateValue<double>(dd,nest::names::C_m,     C_m);
  updateValue<double>(dd,nest::names::g_L,     g_L);

  updateValue<double>(dd,nest::names::tau_syn_ex, tau_synE);
  updateValue<double>(dd,nest::names::tau_syn_in, tau_synI);

  updateValue<double>(dd,nest::names::I_e,     I_e);
  
  updateValue<nest::double_t>(dd, "tau_j",      tau_j);
  updateValue<nest::double_t>(dd, "tau_e",      tau_e);
  updateValue<nest::double_t>(dd, "tau_p",      tau_p);
  updateValue<nest::double_t>(dd, "kappa",      kappa);
  updateValue<nest::double_t>(dd, "gain",      gain);
  updateValue<nest::double_t>(dd, "bias",      bias);
  updateValue<nest::double_t>(dd, "fmax",      fmax);
  updateValue<nest::double_t>(dd, "epsilon",      epsilon);

  if ( V_reset >= V_th )
    throw nest::BadProperty("Reset potential must be smaller than threshold.");
    
  if ( C_m <= 0 )
    throw nest::BadProperty("Capacitance must be strictly positive.");
    
  if ( t_ref < 0 )
    throw nest::BadProperty("Refractory time cannot be negative.");
      
  if ( tau_synE <= 0 || tau_synI <= 0 )
    throw nest::BadProperty("All time constants must be strictly positive.");
}

void mynest::iaf_cond_alpha_bias::State_::get(DictionaryDatum &dd) const
{
  def<double>(dd, nest::names::V_m, y[V_M]); // Membrane potential
}

void mynest::iaf_cond_alpha_bias::State_::set(const DictionaryDatum& dd, const Parameters_&)
{
  updateValue<double>(dd, nest::names::V_m, y[V_M]);
}


/* ---------------------------------------------------------------- 
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */

mynest::iaf_cond_alpha_bias::iaf_cond_alpha_bias()
  : Archiving_Node(), 
    P_(), 
    S_(P_),
    B_(*this)
{
  recordablesMap_.create();
}

mynest::iaf_cond_alpha_bias::iaf_cond_alpha_bias(const iaf_cond_alpha_bias& n)
  : Archiving_Node(n), 
    P_(n.P_), 
    S_(n.S_),
    B_(n.B_, *this)
{
}

mynest::iaf_cond_alpha_bias::~iaf_cond_alpha_bias()
{
  // GSL structs only allocated by init_nodes_(), so we need to protect destruction
  if ( B_.s_ ) gsl_odeiv_step_free(B_.s_);
  if ( B_.c_ ) gsl_odeiv_control_free(B_.c_);
  if ( B_.e_ ) gsl_odeiv_evolve_free(B_.e_);
}

/* ---------------------------------------------------------------- 
 * Node initialization functions
 * ---------------------------------------------------------------- */

void mynest::iaf_cond_alpha_bias::init_node_(const Node& proto)
{
  const iaf_cond_alpha_bias& pr = downcast<iaf_cond_alpha_bias>(proto);
  P_ = pr.P_;
  S_ = pr.S_;
}

void mynest::iaf_cond_alpha_bias::init_state_(const Node& proto)
{
  const iaf_cond_alpha_bias& pr = downcast<iaf_cond_alpha_bias>(proto);
  S_ = pr.S_;
}

void mynest::iaf_cond_alpha_bias::init_buffers_()
{
  Archiving_Node::clear_history();

  B_.spike_exc_.clear();       // includes resize
  B_.spike_inh_.clear();       // includes resize
  B_.currents_.clear();        // includes resize

  B_.logger_.reset();

  B_.step_ = nest::Time::get_resolution().get_ms();
  B_.IntegrationStep_ = B_.step_;

  static const gsl_odeiv_step_type* T1 = gsl_odeiv_step_rkf45;
  
  if ( B_.s_ == 0 )
    B_.s_ = gsl_odeiv_step_alloc (T1, State_::STATE_VEC_SIZE);
  else 
    gsl_odeiv_step_reset(B_.s_);
    
  if ( B_.c_ == 0 )  
    B_.c_ = gsl_odeiv_control_y_new (1e-3, 0.0);
  else
    gsl_odeiv_control_init(B_.c_, 1e-3, 0.0, 1.0, 0.0);
    
  if ( B_.e_ == 0 )  
    B_.e_ = gsl_odeiv_evolve_alloc(State_::STATE_VEC_SIZE);
  else 
    gsl_odeiv_evolve_reset(B_.e_);
  
  B_.sys_.function  = iaf_cond_alpha_bias_dynamics; 
  B_.sys_.jacobian  = NULL;
  B_.sys_.dimension = State_::STATE_VEC_SIZE;
  B_.sys_.params    = reinterpret_cast<void*>(this);

  B_.I_stim_ = 0.0;
}

void mynest::iaf_cond_alpha_bias::calibrate()
{
  B_.logger_.init();  // ensures initialization in case mm connected after Simulate

  V_.PSConInit_E  = 1.0 * numerics::e / P_.tau_synE;
  V_.PSConInit_I  = 1.0 * numerics::e / P_.tau_synI;
  V_.RefractoryCounts = nest::Time(nest::Time::ms(P_.t_ref)).get_steps();
  
  assert(V_.RefractoryCounts >= 0);  // since t_ref >= 0, this can only fail in error
}

/* ---------------------------------------------------------------- 
 * Update and spike handling functions
 * ---------------------------------------------------------------- */

void mynest::iaf_cond_alpha_bias::update(nest::Time const & origin, const nest::long_t from, const nest::long_t to)
{
   
  assert(to >= 0 && (nest::delay) from < nest::Scheduler::get_min_delay());
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
        throw nest::GSLSolverFailure(get_name(), status);
    }

    // refractoriness and spike generation 
    if ( S_.r )
    {// neuron is absolute refractory
	    --S_.r;
	    S_.y[State_::V_M] = P_.V_reset;  // clamp potential
    }
    else
      // neuron is not absolute refractory
      if ( S_.y[State_::V_M] >= P_.V_th )
      {
	S_.r              = V_.RefractoryCounts;
	S_.y[State_::V_M] = P_.V_reset;

        S_.y[State_::Z_J] += (1000.0/(P_.fmax*B_.step_) - S_.y[State_::Z_J] + P_.epsilon) * B_.step_ / P_.tau_j; /* 10k = 1000 * 10 timesteps... */
        S_.y[State_::E_J] += (S_.y[State_::Z_J] - S_.y[State_::E_J]) * B_.step_ / P_.tau_e;
        S_.y[State_::P_J] += P_.kappa * (S_.y[State_::E_J] - S_.y[State_::P_J]) * B_.step_ / P_.tau_p;


	// log spike with Archiving_Node
	set_spiketime(nest::Time::step(origin.get_steps()+lag+1));
	
	nest::SpikeEvent se;
	network()->send(*this, se, lag);
      }

    // add incoming spikes
    S_.y[State_::DG_EXC] += B_.spike_exc_.get_value(lag) * V_.PSConInit_E;
    S_.y[State_::DG_INH] += B_.spike_inh_.get_value(lag) * V_.PSConInit_I;

    S_.bias = P_.gain * std::log(S_.y[State_::P_J]);

    // set new input current
    B_.I_stim_ = B_.currents_.get_value(lag);
   
    // log state data
    B_.logger_.record_data(origin.get_steps() + lag);
  }
}

void mynest::iaf_cond_alpha_bias::handle(nest::SpikeEvent & e)
{
  assert(e.get_delay() > 0);

  if(e.get_weight() > 0.0)
    B_.spike_exc_.add_value(e.get_rel_delivery_steps(network()->get_slice_origin()),
			    e.get_weight() * e.get_multiplicity() );
  else
    B_.spike_inh_.add_value(e.get_rel_delivery_steps(network()->get_slice_origin()),
			 -e.get_weight() * e.get_multiplicity() );  // ensure conductance is positive
}

void mynest::iaf_cond_alpha_bias::handle(nest::CurrentEvent& e)
{
  assert(e.get_delay() > 0);

  // add weighted current; HEP 2002-10-04
  B_.currents_.add_value(e.get_rel_delivery_steps(network()->get_slice_origin()), 
			 e.get_weight() * e.get_current() );
}

void mynest::iaf_cond_alpha_bias::handle(nest::DataLoggingRequest& e)
{
  B_.logger_.handle(e);
}

#endif //HAVE_GSL
