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
	void fill_post_spiketimes(nest::double_t t_spike,
			std::vector<nest::double_t>& post_spiketimes,
			nest::int_t BUFFER);

	void progress_state_variables(nest::double_t t_spike,
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
};

inline void BCPNNDopaConnection::update_dopamine_(
		const nest::vector<nest::spikecounter>& dopa_spikes) {
	double_t minus_dt = dopa_spikes[dopa_spikes_idx_].spike_time_
			- dopa_spikes[dopa_spikes_idx_ + 1].spike_time_;
	++dopa_spikes_idx_;
	n_ = n_ * std::exp(minus_dt / tau_n_)
	+ dopa_spikes[dopa_spikes_idx_].multiplicity_ / tau_n_;
}

inline void BCPNNDopaConnection::update_K_(double_t c0, double_t n0,
		double_t minus_dt) {
	K_ = K_ + n0 / taue_ - b_ / taue_;
	if (K_ < Kmin_)
		K_ = Kmin_;

	if (K_ > Kmax_)
		K_ = Kmax_;
}
inline void BCPNNDopaConnection::process_dopa_spikes_(
		const nest::vector<nest::spikecounter>& dopa_spikes, double_t t0,
		double_t t1) {
	if ((dopa_spikes.size() > dopa_spikes_idx_ + 1)
			&& (dopa_spikes[dopa_spikes_idx_ + 1].spike_time_ <= t1)) {
		double_t n0 = n_
				* std::exp(
						(dopa_spikes[dopa_spikes_idx_].spike_time_ - t0)
						/ tau_n_);
		update_K_(c_, n0, t0 - dopa_spikes[dopa_spikes_idx_ + 1].spike_time_);
		update_dopamine_(dopa_spikes);
		double_t cd;
		while ((dopa_spikes.size() > dopa_spikes_idx_ + 1)
				&& (dopa_spikes[dopa_spikes_idx_ + 1].spike_time_ <= t1)) {
			cd = c_
					* std::exp(
							(t0 - dopa_spikes[dopa_spikes_idx_].spike_time_)
							/ tau_c_);
			update_K_(cd, n_,
					dopa_spikes[dopa_spikes_idx_].spike_time_
					- dopa_spikes[dopa_spikes_idx_ + 1].spike_time_);
			update_dopamine_(dopa_spikes);
		}
		cd = c_
				* std::exp(
						(t0 - dopa_spikes[dopa_spikes_idx_].spike_time_)
						/ tau_c_);
		update_K_(cd, n_, dopa_spikes[dopa_spikes_idx_].spike_time_ - t1);
	} else {
		double_t n0 = n_
				* std::exp(
						(dopa_spikes[dopa_spikes_idx_].spike_time_ - t0)
						/ tau_n_);
		update_K_(c_, n0, t0 - t1);
	}
	c_ = c_ * std::exp((t0 - t1) / tau_c_);
}

inline void BCPNNDopaConnection::hebbian_learning(double_t resolution) {
	pi_ += K_ * (ei_ - pi_) * resolution / taup_;
	pj_ += K_ * (ej_ - pj_) * resolution / taup_;
	pij_ += K_ * (eij_ - pij_) * resolution / taup_;
}

inline void BCPNNDopaConnection::anti_hebbian_learning(double_t resolution) {
	pi_ += K_ * (ei_ - pi_) * resolution / taup_;
	pj_ += K_ * (ej_ - pj_) * resolution / taup_;
	pij_ += K_ * (eij_ - pij_) * resolution / taup_;
}

inline void BCPNNDopaConnection::check_connection(nest::Node& s, nest::Node& r,
		nest::port receptor_type, nest::double_t t_lastspike) {
	nest::ConnectionHetWD::check_connection(s, r, receptor_type, t_lastspike);
	r.register_stdp_connection(
			t_lastspike - nest::Time(nest::Time::step(delay_)).get_ms());
}

inline
void BCPNNDopaConnection::fill_post_spiketimes(nest::double_t t_spike,
		std::vector<nest::double_t>& post_spiketimes,
		nest::int_t BUFFER) {

	nest::int_t counter = 0;
	nest::double_t dendritic_delay =
			nest::Time(nest::Time::step(delay_)).get_ms();
	std::deque<nest::histentry>::iterator start;
	std::deque<nest::histentry>::iterator finish;

	target_->get_history(t_last_update_ - dendritic_delay,
			t_spike - dendritic_delay, &start, &finish);

	nest::double_t t0 = t_last_update_;
	nest::double_t minus_dt;

	while (start != finish) {
		t0 = start->t_ + dendritic_delay;
		minus_dt = t_last_update_ - t0;
		post_spiketimes.at(counter) = start->t_;
		start++;
		if (minus_dt == 0) {
			continue;
		}
		counter++;
		if (counter >= BUFFER) {
			BUFFER = 2 * BUFFER;
			post_spiketimes.resize(BUFFER);
		}
	}

}

inline void BCPNNDopaConnection::progress_state_variables(nest::double_t1,
		std::vector<nest::double_t>& post_spiketimes) {

	const nest::vector<nest::spikecounter>& dopa_spikes = vt_->deliver_spikes();
	nest::double_t resolution = nest::Time::get_resolution().get_ms();
	nest::int_t spike_width =  (nest::int_t)(1 / resolution);
	nest::double_t spike_height = 1000.0 / fmax_;

	nest::double_t n0;
	nest::int_t counter = 0;
	nest::int_t number_iterations = (nest::int_t)(
			(((t1 - t_last_update_) / resolution)));
	nest::int_t j_flag = 0;
	nest::int_t j_counter = 1;

	//for (nest::int_t timestep = 0; timestep < number_iterations; timestep++) {
	for (nest::int_t t = t_last_update_; t < t1; t=t+resolution) {

		if ((dopa_spikes_idx_ < dopa_spikes.size()) &&
				(resolution/2. <= std::abs(dopa_spikes[dopa_spikes_idx_].spike_time_ )-t)){
			n0 = dopa_spikes[dopa_spikes_idx_].multiplicity_ * spike_width;
			dopa_spikes_idx_++;
		} else {
			n0=0;
		}

		n_ = n_ * std::exp(-resolution / tau_n_) + n0 / tau_n_;

		yi_ = 0.0;
		yj_ = 0.0;
		if (t== t_last_update_
				&& (nest::int_t)((t_last_update_)) != 0) {
			yi_ = 1.0*spike_width*spike_height;
		}
		if ((resolution/2.<=std::abs((post_spiketimes.at(counter) - t))
								&& (nest::int_t)(((post_spiketimes.at(counter)))) != 0)
								|| (j_flag == 1)) {
			yj_ = 1.0*spike_width*spike_height;
//			if (j_counter != spike_width) {
//				j_counter++;
//				j_flag = 1;
//			} else {
//				counter++;
//				j_flag = 0;
//				j_counter = 1;
//			}
		}

		zi_ += ( yi_ - zi_ + epsilon_) * resolution / taui_;
		zj_ += ( yj_ - zj_ + epsilon_) * resolution / tauj_;
		ei_ += (zi_ - ei_) * resolution / taue_;
		ej_ += (zj_ - ej_) * resolution / taue_;
		eij_ += (zi_ * zj_ - eij_) * resolution / taue_;

		if (dopamine_modulated_) {
			if (K_ * reverse_ >= 0) {
				hebbian_learning(resolution);
			} else {
			}
		} else {
			pi_ += (ei_ - pi_) * resolution / taup_;
			pj_ += (ej_ - pj_) * resolution / taup_;
			pij_ += (eij_ - pij_) * resolution / taup_;
		};
	}
	bias_ = std::log(pj_);
	weight_ = gain_ * (std::log(pij_ / (pi_ * pj_)));
}

inline void BCPNNDopaConnection::send(nest::Event& e, nest::double_t,
		const nest::CommonSynapseProperties&) {
	nest::double_t t_spike = e.get_stamp().get_ms();
	nest::int_t BUFFER = 20;
	std::vector < nest::double_t > post_spiketimes(BUFFER);

	fill_post_spiketimes(t_spike, post_spiketimes, BUFFER);

	progress_state_variables(t_spike, post_spiketimes);

	t_last_update_ = t_spike;



	e.set_receiver(*target_);
	e.set_weight(weight_);
	e.set_delay(delay_);
	e.set_rport(rport_);
	e();
	post_spiketimes.clear();
}

inline void BCPNNDopaConnection::trigger_update_weight(
		const nest::vector<nest::spikecounter>& dopa_spikes,
		const nest::double_t t_trig, const nest::CommonSynapseProperties&) {

	nest::int_t BUFFER = 20;
	std::vector < nest::double_t > post_spiketimes(BUFFER);

	fill_post_spiketimes(t_trig, post_spiketimes, BUFFER);

	progress_state_variables(t_trig, post_spiketimes);

	t_last_update_ = t_trig;

	// Since dopa_spikes arrives as avariable
	dopa_spikes_idx_ = 0;
}


} /* of namespace mynest */
#endif /* of #ifndef BCPNN_CONNECTION_DOPAMINE_H */

