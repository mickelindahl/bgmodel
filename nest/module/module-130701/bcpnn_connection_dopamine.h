/*
 *  bcpnn_connection.h
 *
 *  Written by  Mikael Lindahl
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
  Author: Mikael Lindahl
          lindahlm@csc.kth.se
  SeeAlso: bcpnn_synapse, synapsedict, stdp_synapse, tsodyks_synapse,
  	  	   static_synapse
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

/**
 * Class containing the common properties for all synapses of type dopamine
 * connection. It subclassed CommonSynapseProperties which a belive is
 * called when a synapse model is used in connection setup. Important is
 * that get node is called early on to set the targets of the synaose
 * for the volume transmitter to collect spikes from.
 */
class BCPNNDopaCommonProperties : public nest::CommonSynapseProperties
{

	friend class BCPNNDopaConnection;

public:

	/**
	 * Default constructor.
	 * Sets all property values to defaults.
	 */
	BCPNNDopaCommonProperties();

	/**
	 * Get all properties and put them into a dictionary.
	 */
	void get_status(DictionaryDatum& d) const;

	/**
	 * Set properties from the values given in dictionary.
	 */
	void set_status(const DictionaryDatum& d, nest::ConnectorModel& cm);

	// overloaded for all supported event types
	void check_event(nest::SpikeEvent&) {}

	nest::Node* get_node();

private:

	nest::double_t b_;
	bool dopamine_modulated_;
	nest::double_t epsilon_;
	nest::double_t fmax_;
	nest::double_t gain_;
	nest::double_t gain_dopa_;
	nest::double_t k_pow_;
	nest::double_t K_;
	nest::double_t tau_n_;
	nest::double_t taui_;
	nest::double_t tauj_;
	nest::double_t taue_;
	nest::double_t taup_;
	nest::double_t reverse_;
	bool sigmoid_;
	nest::double_t sigmoid_mean_;
	nest::double_t sigmoid_slope_;
	nest::volume_transmitter* vt_;

};



class BCPNNDopaConnection : public nest::ConnectionHetWD
{



public:
	/**
	 * Default Constructor.
	 * Sets default values for all parameters. Needed by GenericConnectorModel.
	 */
	BCPNNDopaConnection();

	/**
	 * Copy constructor from a property object.
	 * Needs to be defined properly in order for GenericConnector to work.
	 */
	BCPNNDopaConnection(const BCPNNDopaConnection&);

	/**
	 * Default Destructor.
	 */
	virtual ~BCPNNDopaConnection() {}


	// Import overloaded virtual function set to local scope.
	using nest::Connection::check_event;

	/*
	 * This function calls check_connection on the sender and checks if the receiver
	 * accepts the event type and receptor type requested by the sender.
	 * Node::check_connection() will either confirm the receiver port by returning
	 * true or false if the connection should be ignored.
	 * We have to override the base class' implementation, since for BCPNNDopa
	 * connections we have to call register_dopamine_connection on the target neuron
	 * to inform the Archiver to collect spikes for this connection.
	 *
	 * \param s The source node
	 * \param r The target node
	 * \param receptor_type The ID of the requested receptor type
	 */
	void check_connection(nest::Node& s, nest::Node& r,
			nest::port receptor_type, nest::double_t t_lastspike);

	/**
	 * Get all properties of this connection and put them into a dictionary.
	 */
	void get_status(DictionaryDatum& d) const;

	/**
	 * Set properties of this connection from the values given in dictionary.
	 */
	void set_status(const DictionaryDatum& d, nest::ConnectorModel& cm);


	/**
	 * Set properties of this connection from position p in the properties
	 * array given in dictionary.
	 */
	void set_status(const DictionaryDatum& d, nest::index p,
			nest::ConnectorModel& cm);

	/**
	 * Create new empty arrays for the properties of this connection in the given
	 * dictionary. It is assumed that they are not existing before.
	 */
	void initialize_property_arrays(DictionaryDatum& d) const;


	/**
	 * Append properties of this connection to the given dictionary. If the
	 * dictionary is empty, new arrays are created first.
	 */
	void append_properties(DictionaryDatum& d) const;

	// overloaded for all supported event types
	void check_event(nest::SpikeEvent&) {}

	/**
	 * Send an event to the receiver of this connection.
	 * \param e The event to send
	 */
	void send(nest::Event& e, nest::double_t t_lastspike,
			const BCPNNDopaCommonProperties& cp);

	void trigger_update_weight(
			const nest::vector<nest::spikecounter>& dopa_spikes,
			const nest::double_t t_trig, const BCPNNDopaCommonProperties&);
private:

	void fill_post_spiketimes(nest::double_t t0, nest::double_t t1);

	void progress_state_variables(
			const nest::vector<nest::spikecounter>& dopa_spikes,
			nest::double_t t0, nest::double_t t1, bool pre_spike,
			const BCPNNDopaCommonProperties& cp);

	nest::double_t bias_;
	nest::double_t yi_;
	nest::double_t yj_;
	nest::double_t zi_;
	nest::double_t zj_;
	nest::double_t zj_c_;
	nest::double_t ei_;
	nest::double_t ej_;
	nest::double_t eij_;
	nest::double_t ej_c_;
	nest::double_t eij_c_;
	nest::double_t pi_;
	nest::double_t pj_;
	nest::double_t pij_;
	nest::double_t k_;
	nest::double_t k_filtered_;
	nest::double_t n_;
	nest::double_t n_add_;
	nest::double_t m_;

	/*
	 dopa_spikes_idx_ refers to the dopamine spike that has just been processes
     after trigger_update_weight a pseudo dopamine spike at t_trig is
     stored at index 0 and dopa_spike_idx_ = 0
	 */
	nest::index dopa_spikes_idx_;

	/*time of last update, which is either time of last presyn. spike or
	 * time-driven update*/

	nest::double_t t_last_update_;

	nest::index spike_idx_;

	nest::vector<nest::double_t> post_spiketimes_;
	nest::int_t BUFFER_;
};


inline void BCPNNDopaConnection::check_connection(nest::Node& s, nest::Node& r,
		nest::port receptor_type, nest::double_t t_lastspike) {
	nest::ConnectionHetWD::check_connection(s, r, receptor_type, t_lastspike);
	r.register_stdp_connection(
			t_lastspike - nest::Time(nest::Time::step(delay_)).get_ms());
}

inline
void BCPNNDopaConnection::fill_post_spiketimes(nest::double_t t0,
		nest::double_t t1) {

	nest::double_t dendritic_delay =
			nest::Time(nest::Time::step(delay_)).get_ms();
	std::deque<nest::histentry>::iterator start;
	std::deque<nest::histentry>::iterator finish;

	target_->get_history(t0 - dendritic_delay,
			t1 - dendritic_delay, &start, &finish);

	while (start != finish) {
		t0 = start->t_ + dendritic_delay;
		post_spiketimes_.push_back(start->t_);
		start++;

		if (spike_idx_ >= BUFFER_) {
			BUFFER_ = 2 * BUFFER_;
			post_spiketimes_.reserve(2 * BUFFER_);
		}
	}
}

inline void BCPNNDopaConnection::progress_state_variables(
		const nest::vector<nest::spikecounter>& dopa_spikes,
		nest::double_t t0,
		nest::double_t t1,
		bool pre_spike,
		const BCPNNDopaCommonProperties& cp) {

	nest::double_t resolution = nest::Time::get_resolution().get_ms();
	nest::double_t spk_amp= 1000.0 / cp.fmax_/resolution;

	nest::double_t ej;
	nest::double_t eij;
	nest::double_t k;

	/*Since post_spiketimes is a variable*/
	spike_idx_ = 0;

	for (nest::double_t t = t0; t<t1;t=t+resolution) {

		/*plus one since dopa_spikes contains psuedo spike. Se evolume
		transmitter.*/
		if ((dopa_spikes_idx_ +1< dopa_spikes.size()) &&
				(resolution/2.
						<= std::abs(dopa_spikes[dopa_spikes_idx_+1].spike_time_ )-t)){

			dopa_spikes_idx_++;
			n_add_ = dopa_spikes[dopa_spikes_idx_].multiplicity_;

		} else {
			n_add_=0;
		}

		/*Update dopamine trace an relative dopamine trace*/
		n_ = n_ * std::exp(-resolution / cp.tau_n_) + n_add_ / cp.tau_n_;
		m_ = n_ + cp.b_;

		if (cp.dopamine_modulated_) {
			k_=cp.gain_dopa_* m_, cp.k_pow_;
   		} else {
			k_=cp.K_;
		}

		yi_ = 0.0;
		yj_ = 0.0;

		if ((t == t_last_update_) && pre_spike) {
			yi_ = spk_amp;
		}
		if  (post_spiketimes_.size()>spike_idx_) {
			yj_ = spk_amp;
			spike_idx_++;
		}

		zi_ += ( yi_ - zi_ + cp.epsilon_) * resolution / cp.taui_;
		zj_ += ( yj_ - zj_ + cp.epsilon_) * resolution / cp.tauj_;
		zj_c_ += ( 1-yj_ - zj_c_ + cp.epsilon_) * resolution / cp.tauj_;

		ei_ += (zi_ - ei_) * resolution / cp.taue_;
		ej_ += (zj_ - ej_) * resolution / cp.taue_;
		eij_+= (zi_ * zj_ - eij_) * resolution / cp.taue_;
		ej_c_ += (zj_c_ - ej_c_) * resolution / cp.taue_;
		eij_c_+= (zi_ * zj_c_ - eij_c_) * resolution / cp.taue_;

		if (cp.reverse_*k_>=0) {
			ej=ej_;
			eij=eij_;
		} else {
			ej=ej_c_;
			eij=eij_c_;
		}


		k_filtered_=std::abs(std::pow(k_, cp.k_pow_));
		if (cp.sigmoid_) {
			k_filtered_=1./(1.+std::exp(-cp.sigmoid_slope_*(k-cp.sigmoid_mean_)));
		}

		pi_ += k_filtered_*(ei_ - pi_) * resolution / cp.taup_;
		pj_ += k_filtered_*(ej - pj_) * resolution / cp.taup_;
		pij_+= k_filtered_*(eij - pij_) * resolution / cp.taup_;

	};


	/*cout << 200 << endl;*/

	bias_ = std::log(pj_);

	nest::double_t w =pij_ / (pi_ * pj_);

	weight_ = cp.gain_ * ((w<0) ? std::log(0.0001) : std::log(w));

	/*Reset variables*/
	post_spiketimes_.clear();
	BUFFER_=20;
}

inline void BCPNNDopaConnection::send(nest::Event& e, nest::double_t,
		const BCPNNDopaCommonProperties& cp) {




    nest::double_t dendritic_delay = nest::Time(nest::Time::step(delay_)).get_ms();
	nest::double_t t_spike = e.get_stamp().get_ms();//+dendritic_delay;

	const vector<nest::spikecounter>& dopa_spikes = cp.vt_->deliver_spikes();

	post_spiketimes_.reserve(BUFFER_);

	fill_post_spiketimes(t_last_update_, t_spike);

	progress_state_variables(dopa_spikes, t_last_update_, t_spike, true, cp);

	t_last_update_ = t_spike;

	e.set_receiver(*target_);
	e.set_weight(weight_);
	e.set_delay(delay_);
	e.set_rport(rport_);
	e();

}

inline void BCPNNDopaConnection::trigger_update_weight(
		const nest::vector<nest::spikecounter>& dopa_spikes,
		const nest::double_t t_trig,
		const BCPNNDopaCommonProperties& cp) {

	post_spiketimes_.reserve(BUFFER_);

	/*
	Minus one since trigger_update_weight is called by the scheduler every
	milisecond, and is called before spike event are sent out.

	Todo: Find a better way to handle this.Alternative way is to make
	the integration so that is can go both back and fourth in time. That is
	when t0 is bigger thatn t1, which happens at send spike,then the integration
	could progress back. This is how the lacey update works in stdp_dopamine
	synapse.
	*/
	fill_post_spiketimes(t_last_update_, t_trig-1);
	progress_state_variables(dopa_spikes, t_last_update_, t_trig-1, false, cp);

	t_last_update_ = t_trig-1;

	// Since dopa_spikes arrives as a variable
	dopa_spikes_idx_ = 0;
}


} /* of namespace mynest */
#endif /* of #ifndef BCPNN_CONNECTION_DOPAMINE_H */

