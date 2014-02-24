/*
 *  bcpnn_connection.h
 *
 *  Written by Philip Tully and Mikae Lindahl
 *
 */

#ifndef BCPNN_CONNECTION_DOPAMINE_H
#define BCPNN_CONNECTION_DOPAMINE_H

/* BeginDocumentation
  Name: bcpnn_synapse - Synapse type for incremental, Bayesian spike-timing 
   dependent plasticity.

  Description:
   bcpnn_synapse is a connector to create synapses with incremental, Bayesian 
   spike timing dependent plasticity.

   tau_i	double - Primary trace presynaptic time constant
   tau_j	double - Primary trace postsynaptic time constant
   tau_e	double - Secondary trace time constant
   tau_p	double - Tertiarty trace time constant
   p_i		double - \
   p_j		double -  >- these 3 initial conditions determine weight, i.e. log(p_ij/(p_i * p_j)).
   p_ij		double - /
   K_		double - Print-now signal // Neuromodulation. Turn off learning, K = 0.
   fmax_        double - Frequency assumed as maximum firing, for match with abstract rule
   epsilon_     double - lowest possible probability of spiking, e.g. lowest assumed firing rate
   bias_        double - ANN interpretation. Only calculated here to demonstrate match to rule. 
                         Will be eliminated in future versions, where bias will be calculated postsynaptically
   gain_    double - Coefficient to scale weight as conductance, can be zero-ed out

  Transmits: SpikeEvent

  References:
   [1] Wahlgren and Lansner (2001) Biological Evaluation of a Hebbian-Bayesian
       learning rule. Neurocomputing, 38-40, 433-438

   [2] Bergel, Transforming the BCPNN Learning Rule for Spiking Units to a
       Learning Rule for Non-Spiking Units (2010). KTH Masters Thesis.

  FirstVersion: November 2011
  CurrentVersion: March 2012
  Author: Philip Tully
          tully@csc.kth.se
  SeeAlso: synapsedict, stdp_synapse, tsodyks_synapse, static_synapse
 */

/* for Debugging */
#include <iostream>
using namespace std;

#include "connection_het_wd.h"
#include "archiving_node.h"
#include "generic_connector.h"
#include <cmath>


#include "numerics.h"
#include "spikecounter.h"
#include "volume_transmitter.h"

namespace mynest
{


class BCPNNDopaConnection : public nest::ConnectionHetWD
{



public:
	BCPNNDopaConnection();
	BCPNNDopaConnection(const BCPNNDopaConnection&);

	~BCPNNDopaConnection() {
	}

	void check_connection(nest::Node& s, nest::Node& r,
			nest::port receptor_type, nest::double_t t_lastspike);
	void get_status(DictionaryDatum& d) const;
	void set_status(const DictionaryDatum& d, nest::ConnectorModel& cm);
	void set_status(const DictionaryDatum& d, nest::index p,
			nest::ConnectorModel& cm);
	void initialize_property_arrays(DictionaryDatum& d) const;
	void append_properties(DictionaryDatum& d) const;
	void send(nest::Event& e, nest::double_t t_lastspike,
			const nest::CommonSynapseProperties& cp);
	using nest::Connection::check_event;

	void check_event(nest::SpikeEvent&) {
	}

	void trigger_update_weight(
			const nest::vector<nest::spikecounter>& dopa_spikes,
			double_t t_trig, const nest::CommonSynapseProperties& cp);

private:
	void update_dopamine_(const nest::vector<nest::spikecounter>& dopa_spikes);
	void update_K_(double_t c0, double_t n0, double_t minus_dt);
	void process_dopa_spikes_(
			const nest::vector<nest::spikecounter>& dopa_spikes, double_t t0,
			double_t t1);
	void hebbian_learning(double_t resolution);
	void anti_hebbian_learning(double_t resolution);
	void fill_post_spiketimes(nest::double_t dendritic_delay,
			nest::double_t& t_spike,
			std::vector<nest::double_t>& post_spiketimes, nest::int_t BUFFER);
	void progress_state_variables(nest::double_t& t_spike,
			std::vector<nest::double_t>& post_spiketimes);
	nest::double_t yi_;
	nest::double_t yj_;
	nest::double_t taui_;
	nest::double_t tauj_;
	nest::double_t taue_;
	nest::double_t taup_;
	nest::double_t epsilon_;
	nest::double_t K_;
	nest::double_t bias_;
	nest::double_t fmax_;
	nest::double_t gain_;
	nest::double_t zi_;
	nest::double_t zj_;
	nest::double_t ei_;
	nest::double_t ej_;
	nest::double_t eij_;
	nest::double_t pi_;
	nest::double_t pj_;
	nest::double_t pij_;
	nest::double_t b_;
	nest::double_t Kmin_;
	nest::double_t Kmax_;
	nest::double_t reverse_;
	nest::double_t tau_c_;
	nest::double_t tau_n_;
	nest::double_t c_;
	nest::double_t n_;
	nest::index dopa_spikes_idx_;
	nest::double_t t_last_update_;
	nest::volume_transmitter* vt_;

	bool dopamine_modulated_;

}; /* of class BCPNNDopaConnection */


inline
void BCPNNDopaConnection::update_dopamine_(const nest::vector<nest::spikecounter>& dopa_spikes)
{
	double_t minus_dt = dopa_spikes[dopa_spikes_idx_].spike_time_ - dopa_spikes[dopa_spikes_idx_+1].spike_time_;
	++dopa_spikes_idx_;
	n_ = n_ * std::exp( minus_dt / tau_n_ ) + dopa_spikes[dopa_spikes_idx_].multiplicity_ / tau_n_;


}

inline
void BCPNNDopaConnection::update_K_(double_t c0, double_t n0, double_t minus_dt)
{
	//double_t taus_ = ( tau_c_ + tau_n_ ) / ( tau_c_ * tau_n_ );
	//K_ = K_ - c0 * ( n0 / taus_ * numerics::expm1( taus_ * minus_dt )
	//- b_ * tau_c_ * numerics::expm1( minus_dt / tau_c_ ) );
	K_= K_+n0/taue_-b_/taue_;

	if ( K_ < Kmin_)
		K_ = Kmin_;
	if ( K_ > Kmax_ )
		K_ = Kmax_;
}

inline
void BCPNNDopaConnection::process_dopa_spikes_(const nest::vector<nest::spikecounter>& dopa_spikes,
		double_t t0, double_t t1)
{
	// process dopa spikes in (t0, t1]
	// propagate weight from t0 to t1
	if ( ( dopa_spikes.size() > dopa_spikes_idx_+1 ) \
			&& ( dopa_spikes[dopa_spikes_idx_+1].spike_time_ <= t1 ) )
	{
		// there is at least 1 dopa spike in (t0, t1]
		// propagate weight up to first dopa spike and update dopamine trace
		// weight and eligibility c are at time t0 but dopamine trace n is at time of last dopa spike
		double_t n0 = n_ * std::exp( ( dopa_spikes[dopa_spikes_idx_].spike_time_ - t0 ) / tau_n_ );  // dopamine trace n at time t0
		update_K_(c_, n0, t0 - dopa_spikes[dopa_spikes_idx_+1].spike_time_);
		update_dopamine_(dopa_spikes);

		// process remaining dopa spikes in (t0, t1]
		double_t cd;
		while ( ( dopa_spikes.size() > dopa_spikes_idx_+1 ) \
				&& ( dopa_spikes[dopa_spikes_idx_+1].spike_time_ <= t1 ) )
		{
			// propagate weight up to next dopa spike and update dopamine trace
			// weight and dopamine trace n are at time of last dopa spike td but eligibility c is at time t0
			cd = c_ * std::exp( ( t0 - dopa_spikes[dopa_spikes_idx_].spike_time_ ) / tau_c_ );  // eligibility c at time of td
			update_K_(cd, n_, dopa_spikes[dopa_spikes_idx_].spike_time_\
					- dopa_spikes[dopa_spikes_idx_+1].spike_time_);
			update_dopamine_(dopa_spikes);
		}

		// propagate weight up to t1
		// weight and dopamine trace n are at time of last dopa spike td but eligibility c is at time t0
		cd = c_ * std::exp( ( t0 - dopa_spikes[dopa_spikes_idx_].spike_time_ ) / tau_c_ );  // eligibility c at time td
		update_K_(cd, n_, dopa_spikes[dopa_spikes_idx_].spike_time_ - t1);
	}
	else
	{
		// no dopamine spikes in (t0, t1]
		// weight and eligibility c are at time t0 but dopamine trace n is at time of last dopa spike
		double_t n0 = n_ * std::exp( ( dopa_spikes[dopa_spikes_idx_].spike_time_ - t0 ) / tau_n_ );  // dopamine trace n at time t0
		update_K_(c_, n0, t0 - t1);
	}

	// update eligibility trace c for interval (t0, t1]
	c_ = c_ * std::exp( ( t0 - t1 ) / tau_c_ );
}

inline
void BCPNNDopaConnection::hebbian_learning(double_t resolution)
{
	pi_  += K_ * (ei_ - pi_) * resolution / taup_/* * eij_*/;
	pj_  += K_ * (ej_ - pj_) * resolution / taup_/* * eij_*/;
	pij_ += K_ * (eij_ - pij_) * resolution / taup_/* * eij_*/;
}

inline
void BCPNNDopaConnection::anti_hebbian_learning(double_t resolution)
{
	//do some magic!
	pi_  += K_ * (ei_ - pi_) * resolution / taup_/* * eij_*/;
	pj_  += K_ * (ej_ - pj_) * resolution / taup_/* * eij_*/;
	pij_ += K_ * (eij_ - pij_) * resolution / taup_/* * eij_*/;
}


inline
void BCPNNDopaConnection::check_connection(nest::Node & s, nest::Node & r,
		nest::port receptor_type, nest::double_t t_lastspike)
{
	nest::ConnectionHetWD::check_connection(s, r, receptor_type, t_lastspike);

	// For a new synapse, t_lastspike contains the point in time of the last spike.
	// So we initially read the history(t_last_spike - dendritic_delay, ...,  T_spike-dendritic_delay]
	// which increases the access counter for these entries.
	// At registration, all entries' access counters of history[0, ..., t_last_spike - dendritic_delay] will be
	// incremented by the following call to Archiving_Node::register_stdp_connection().
	// See bug #218 for details.
	r.register_stdp_connection(t_lastspike - nest::Time(nest::Time::step(delay_)).get_ms());
}

/* Send an event to the receiver of this connection.
 * \param e The event to send
 * \param p The port under which this connection is stored in the Connector.
 * \param t_lastspike Time point of last spike emitted

   note: every time this method is called by an outside function, a presynaptic
       event has occured and is being transmitted to the postsynaptic side. */



inline
void BCPNNDopaConnection::send(nest::Event& e, nest::double_t, const nest::CommonSynapseProperties &)
{
	nest::double_t t_spike = e.get_stamp().get_ms();  /* time stamp of current spike event */
	nest::double_t dendritic_delay = nest::Time(nest::Time::step(delay_)).get_ms();    /* delay from dendrite -> soma */
	nest::double_t resolution = nest::Time::get_resolution().get_ms();               /* nest.GetKernelStatus('resolution') simulation timestep */
	nest::int_t spike_width = int(1/resolution);                     /* assume spike width of 1ms, resolution is 0.1 so mult by 10 */
	nest::double_t spike_height = 1000.0 / fmax_;     /* normalizing to match this spiking rule to abstract = 1000/FMAX (Hz)*/
	nest::int_t BUFFER = 20;                          /* hold postsynaptic time stamps, reallocate as necessary */
	std::vector<nest::double_t> post_spiketimes(BUFFER);

	nest::double_t min_weight = epsilon_/std::pow(0.5 ,2);         /* theoretical minimum weight = epsilon/(0.5*0.5) */


	//NEW
	nest::int_t counter_dopa = 0;                     /* ensuring traces reverberate for duration of the spike width */
	nest::double_t n0;



	// get history of dopamine spikes
	const nest::vector<nest::spikecounter>& dopa_spikes = vt_->deliver_spikes();

	/*STEP ONE: Get all timings of pre and postsynaptic spikes. Post store in dynamically allocated array */
	nest::int_t counter = 0;

	/* get spike history in relevant range (t1, t2] from post-synaptic neuron */
	std::deque<nest::histentry>::iterator start;
	std::deque<nest::histentry>::iterator finish;

	/* Found in nestkernel/archiving_node.
	 * target_ resides in connnection.h and is aprotected Node variable
	 * target_ is used to access get_history function. The possynaptic
	 * spikes are retrieved between t_lastspike and t_spike.
	 * Initially read the history(t_last_spike - dendritic_delay, ...,  T_spike-dendritic_delay] which increases the
       access counter for these entries. At registration, all entries' access counters of history[0, ...,
       t_last_spike - dendritic_delay] have been incremented by Archiving_Node::register_stdp_connection().
       See bug #218 for details. */
	target_->get_history(t_last_update_ - dendritic_delay, t_spike - dendritic_delay, &start, &finish);

	/* For spike order pre-post, if dopamine present facilitate else depress.
       Pre  spikes: |       |  t_lastpike is the last pre spike and t_spike is the current pre spike
       Post spikes    | ||		 start is a pointer to the first post spike in the interval between the
       two pre spikes. It is then iterated until the last post spike in the interval */


	nest::double_t t0 = t_last_update_;
	/* iterate over post spikes between pre spike at time t_last_update up until
	 * pre spike at time t_spike. */

	nest::double_t minus_dt;
                         /* ensuring traces reverberate for duration of the spike width */
	while (start != finish)
	{
		//process_dopa_spikes_(dopa_spikes, t0, start->t_ + dendritic_delay);
		t0 = start->t_ + dendritic_delay;
		minus_dt = t_last_update_ - t0;
		post_spiketimes.at(counter) = start->t_;
		start++;

		/* CASE: BOTH PRE AND POST SPIKE @ LAST TIME STEP. IGNORE. */
		if (minus_dt == 0)
		{
			continue;
		} /* of if */

		counter++;
		if(counter >= BUFFER)     /* reallocate array size if many postsynaptic events occur in window */
		{
			BUFFER = 2 * BUFFER;
			post_spiketimes.resize(BUFFER);
		} /* of if */
	} /* of while */


	/* STEP TWO: Consider the presynaptic firing window, delta t resolution, and update the traces */

	/* nest stores with ms precision the timing of the spike. */
	/* the following loop iterates through the presynaptic spike difference window */
	nest::int_t counter2 = 0;
	//counter = 0;
	nest::int_t number_iterations = (nest::int_t)((t_spike - t_last_update_)/resolution);
	nest::int_t j_flag = 0;
	nest::int_t j_counter = 1;

	for (nest::int_t timestep = 0; timestep < number_iterations; timestep++)
	{
		// Dopamine trace
		if (counter_dopa<dopa_spikes.size())
		{
			if (counter_dopa[counter_dopa]) {
				n0=dopa_spikes[dopa_spikes_idx_].multiplicity_*spike_width;
			}
		}
		n_ = n_ * std::exp( -resolution / tau_n_ ) +n0/tau_n_ ;

		/* CASE: Default. Neither Pre nor Post spike. */
		yi_ = 0.0;
		yj_ = 0.0;

		/* CASE: Pre without (*OR WITH post) spike - synchronous events handled automatically. */
		if(timestep >= 0 && timestep < spike_width && (nest::int_t)t_last_update_ != 0)
		{
			yi_ = 1.0;
		}

		/* CASE: Post spiking */
		if ((timestep == (nest::int_t)((post_spiketimes.at(counter2) \
				- t_last_update_)/resolution) && (nest::int_t)(post_spiketimes.at(counter2)) != 0) \
				|| (j_flag == 1))
		{
			yj_ = 1.0;
			/*counter++;*/
			/**/if (j_counter != spike_width)
			{
				j_counter++;
				j_flag = 1;
			} else {
				counter2++;
				j_flag = 0;
				j_counter = 1;
			}/**/
		}
		//		if (timestep==dopa_spikes[dopa_spikes_idx_]) {
		//		}
	}
	/* Primary synaptic traces. Noise - commented out*/
	zi_ += (spike_height * yi_ - zi_ + epsilon_ /*+ (0.01 + (double)rand() / RAND_MAX * (0.05 - 0.01))*/ ) * resolution / taui_;
	zj_ += (spike_height * yj_ - zj_ + epsilon_ /*+ (0.01 + (double)rand() / RAND_MAX * (0.05 - 0.01))*/ ) * resolution / tauj_;

	/* Secondary synaptic traces */
	ei_  += (zi_ - ei_) * resolution / taue_;
	ej_  += (zj_ - ej_) * resolution / taue_;
	eij_ += (zi_ * zj_ - eij_) * resolution / taue_;


	if (dopamine_modulated_)
	{
		//process_dopa_spikes_(dopa_spikes, t0, t_spike);
		if ( K_*reverse_>=0)
		{
			hebbian_learning(resolution);
		} else {
			//anti_hebbian_learning(resolution);
		}
	} else {

		pi_  += (ei_ - pi_) * resolution / taup_/* * eij_*/;
		pj_  += (ej_ - pj_) * resolution / taup_/* * eij_*/;
		pij_ += (eij_ - pij_) * resolution / taup_/* * eij_*/;
	}

	/*weight_ = gain_ * std::log(pij_ / (pi_ * pj_)) /*- std::log(min_weight)*/;
	/*cout << ei_ << endl;*/
	/* of for */

	//Set t_last update
	t_last_update_= t_spike;

	/* Update the weight & bias before event is sent. Use commented normalization to
       implement soft weight bounds, this way the weight will never go below 0 because
       you push all weights up by the most negative weight possible. */
	bias_ = std::log(pj_);
	weight_ = gain_ * (std::log(pij_ / (pi_ * pj_)) /*- std::log(min_weight) */);

	/* STEP THREE. Implement hard weight bounds. NOTE if using above normalization, weights
                   are soft-bounded above zero already. */
	/*weight_ = (weight_ < 0) ? weight_ : 0.0;
      nest::double_t Wmax = ...;
      weight_ = (weight_ > Wmax) ? weight_ : Wmax;*/

	/* Send the spike to the target */
	e.set_receiver(*target_);
	e.set_weight(weight_);
	e.set_delay(delay_);
	e.set_rport(rport_);
	e();

	/* final clean up */
	post_spiketimes.clear();

} /* of BCPNNDopaConnection::send */

//Used in volume_tansmitter to update weights. In order to push dopamine
//spikes to synapses which have not spiked.
inline
void BCPNNDopaConnection::trigger_update_weight(const nest::vector<nest::spikecounter>& dopa_spikes,
		const nest::double_t t_trig, const nest::CommonSynapseProperties& )
{
	// Propaget y, e, p, and n traces but do not increment the weight since
	// there is no spike at t_trig.

	// propagate all state variables to time t_trig
	// this does not include the depression trace K_minus, which is updated in the postsyn.neuron

	// purely dendritic delay
	double_t dendritic_delay = nest::Time(nest::Time::step(delay_)).get_ms();

	// get spike history in relevant range (t_last_update, t_trig] from postsyn. neuron
	std::deque<nest::histentry>::iterator start;
	std::deque<nest::histentry>::iterator finish;
	target_->get_history(t_last_update_ - dendritic_delay, t_trig - dendritic_delay, &start, &finish);

	// facilitation due to postsyn. spikes since last update
	double_t t0 = t_last_update_;
	double_t minus_dt;
	while ( start != finish )
	{
		process_dopa_spikes_(dopa_spikes, t0, start->t_ + dendritic_delay);
		t0 = start->t_ + dendritic_delay;
		minus_dt = t_last_update_ - t0;
		//facilitate_(Kplus_ * std::exp( minus_dt / cp.tau_plus_ ), cp);
		++start;
	}

	// propagate weight, eligibility trace c, dopamine trace n and facilitation trace K_plus to time t_trig
	// but do increment/decrement as there are no spikes to be handled at t_trig
	process_dopa_spikes_(dopa_spikes, t0, t_trig);
	n_ = n_ * std::exp( ( dopa_spikes[dopa_spikes_idx_].spike_time_ - t_trig ) / tau_n_ );
	//Kplus_ = Kplus_ * std::exp( ( t_last_update_ - t_trig ) / cp.tau_plus_);

	t_last_update_ = t_trig;
	dopa_spikes_idx_ = 0;
}


} /* of namespace mynest */
#endif /* of #ifndef BCPNN_CONNECTION_DOPAMINE_H */

