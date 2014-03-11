/*
 *  bcpnn_connection_dopamine.cpp
 *
 *  Written by Philip Tully and Mkael Lindahl
 *
 */

#include "network.h"
#include "dictdatum.h"
#include "connector_model.h"
#include "common_synapse_properties.h"
#include "bcpnn_connection_dopamine.h"
#include "event.h"



namespace mynest
{

//
// Implementation of class STDPDopaCommonProperties.
//

BCPNNDopaCommonProperties::BCPNNDopaCommonProperties() :
   CommonSynapseProperties(),

   b_(0.0),
   dopamine_modulated_(false),
   epsilon_(0.001),
   fmax_(50.0),
   fmax_anti_(950.0),
   gain_(1.0),
   gain_dopa_(1.0),
   K_(1.0),
   reverse_(1.0),
   tau_n_(200.0),
   taui_(10.0),
   tauj_(10.0),
   taue_(100.0),
   taup_(1000.0),
   vt_(0)



{}


void BCPNNDopaCommonProperties::get_status(DictionaryDatum & d) const
{
	CommonSynapseProperties::get_status(d);
	if(vt_!= 0)
		def<nest::long_t>(d, "vt", vt_->get_gid());
	else
		def<nest::long_t>(d, "vt", -1);

	def<nest::double_t>(d, "b", b_);
	def<bool>(d, "dopamine_modulated", dopamine_modulated_);
	def<nest::double_t>(d, "epsilon", epsilon_);
	def<nest::double_t>(d, "fmax", fmax_);
	def<nest::double_t>(d, "fmax_anti", fmax_anti_);
	def<nest::double_t>(d, "gain", gain_);
	def<nest::double_t>(d, "gain_dopa", gain_dopa_);
	def<nest::double_t>(d, "K", K_);
	def<nest::double_t>(d, "reverse", reverse_);
	def<nest::double_t>(d, "tau_i", taui_);
	def<nest::double_t>(d, "tau_j", tauj_);
	def<nest::double_t>(d, "tau_e", taue_);
	def<nest::double_t>(d, "tau_p", taup_);
	def<nest::double_t>(d, "tau_n", tau_n_);

}

void BCPNNDopaCommonProperties::set_status(const DictionaryDatum & d,
		nest::ConnectorModel &cm)
{
  nest::CommonSynapseProperties::set_status(d, cm);

  nest::long_t vtgid;
  if ( updateValue<nest::long_t>(d, "vt", vtgid) )
  {
    vt_ = dynamic_cast<nest::volume_transmitter *>(
    		nest::NestModule::get_network().get_node(vtgid));

    if(vt_==0)
	throw nest::BadProperty("Dopamine source must be volume transmitter");
  }

  updateValue<nest::double_t>(d, "b", b_);
  updateValue<bool>(d, "dopamine_modulated", dopamine_modulated_);
  updateValue<nest::double_t>(d, "epsilon", epsilon_);
  updateValue<nest::double_t>(d, "fmax", fmax_);
  updateValue<nest::double_t>(d, "fmax_anti", fmax_anti_);
  updateValue<nest::double_t>(d, "gain", gain_);
  updateValue<nest::double_t>(d, "gain_dopa", gain_dopa_);
  updateValue<nest::double_t>(d, "K", K_);
  updateValue<nest::double_t>(d, "reverse", reverse_);
  updateValue<nest::double_t>(d, "tau_i", taui_);
  updateValue<nest::double_t>(d, "tau_j", tauj_);
  updateValue<nest::double_t>(d, "tau_e", taue_);
  updateValue<nest::double_t>(d, "tau_p", taup_);
  updateValue<nest::double_t>(d, "tau_n", tau_n_);
}

nest::Node* BCPNNDopaCommonProperties::get_node()
{
	//std::cout << 12345 << endl;


  if(vt_==0)
    throw nest::BadProperty("No volume transmitter has been assigned to the dopamine synapse.");
  else
    return vt_;
}


//
// Implementation of class BCPNNDopaConnection.
//

BCPNNDopaConnection::BCPNNDopaConnection() :
	bias_(0.0),
	BUFFER_(20),
	dopa_spikes_idx_(0),
	ei_(0.01),
	ej_(0.01),
	eij_(0.001),
	n_(0.0),
	m_(0.0),
	pi_(0.01),
	pj_(0.01),
	pij_(0.001),
	spike_idx_(0),
	t_last_update_(0.0),
	yi_(0.0),              //initial conditions
	yj_(0.0),
	zi_(0.01),
	zj_(0.01)
{}

  BCPNNDopaConnection::BCPNNDopaConnection(const BCPNNDopaConnection &rhs) :
    nest::ConnectionHetWD(rhs)
  {
	bias_ = rhs.bias_;
	BUFFER_ = rhs.BUFFER_;
    dopa_spikes_idx_ = rhs.dopa_spikes_idx_;
    ei_ = rhs.ei_;
    ej_ = rhs.ej_;
    eij_ = rhs.eij_;
    n_ = rhs.n_;
    m_ = rhs.m_;
    pi_ = rhs.pi_;
    pj_ = rhs.pj_;
    pij_ = rhs.pij_;
    post_spiketimes_=rhs.post_spiketimes_;
    spike_idx_=rhs.spike_idx_;
    t_last_update_ = rhs.t_last_update_;
    yi_ = rhs.yi_;
    yj_ = rhs.yj_;
    zi_ = rhs.zi_;
    zj_ = rhs.zj_;


  }

  void BCPNNDopaConnection::get_status(DictionaryDatum & d) const
  {
    nest::ConnectionHetWD::get_status(d);

	def<nest::double_t>(d, "bias", bias_);
    def<nest::double_t>(d, "n", n_);
    def<nest::double_t>(d, "m", m_);
    def<nest::double_t>(d, "p_i", pi_);
    def<nest::double_t>(d, "p_j", pj_);
    def<nest::double_t>(d, "p_ij", pij_);
  }

  void BCPNNDopaConnection::set_status(const DictionaryDatum & d,
		  nest::ConnectorModel &cm)
  {
	// base class properties
	nest::ConnectionHetWD::set_status(d, cm);

    updateValue<nest::double_t>(d, "bias", bias_);
    updateValue<nest::double_t>(d, "n", n_);
    updateValue<nest::double_t>(d, "m", m_);
    updateValue<nest::double_t>(d, "p_i", pi_);
    updateValue<nest::double_t>(d, "p_j", pj_);
    updateValue<nest::double_t>(d, "p_ij", pij_);
  }


  void BCPNNDopaConnection::set_status(const DictionaryDatum & d,
		  nest::index p, nest::ConnectorModel &cm)
  {
	  nest::ConnectionHetWD::set_status(d, p, cm);

	  if ( d->known("ns")       ||
		  d->known("dopamine_modulateds")   ||
		  d->known("epsilons")  ||
		  d->known("fmaxs")     ||
		  d->known("gains")     ||
		  d->known("Ks")        ||
		  d->known("reverses")  ||
		  d->known("tau_i")     ||
		  d->known("tau_j")     ||
		  d->known("tau_e")     ||
		  d->known("tau_p")     ||
		  d->known("tau_n") )

	  {
		  cm.network().message(SLIInterpreter::M_ERROR,
				  "STDPDopaConnection::set_status()",
				  "you are trying to set common properties via an individual synapse.");

	  }


    nest::set_property<nest::double_t>(d, "bias", p, bias_);
    nest::set_property<nest::double_t>(d, "n", p, n_);
    nest::set_property<nest::double_t>(d, "m", p, m_);

    nest::set_property<nest::double_t>(d, "p_i", p, pi_);
    nest::set_property<nest::double_t>(d, "p_j", p, pj_);
    nest::set_property<nest::double_t>(d, "p_ij", p, pij_);
  }

  void BCPNNDopaConnection::initialize_property_arrays(DictionaryDatum & d) const
  {
    nest::ConnectionHetWD::initialize_property_arrays(d);

    initialize_property_array(d, "bias");
    initialize_property_array(d, "n");
    initialize_property_array(d, "m");
    initialize_property_array(d, "p_i");
    initialize_property_array(d, "p_j");
    initialize_property_array(d, "p_ij");
  }

  /**
   * Append properties of this connection to the given dictionary. If the
   * dictionary is empty, new arrays are created first.
   */
  void BCPNNDopaConnection::append_properties(DictionaryDatum & d) const
  {
    nest::ConnectionHetWD::append_properties(d);

    append_property<nest::double_t>(d, "bias", bias_);
    append_property<nest::double_t>(d, "n", n_);
    append_property<nest::double_t>(d, "m", m_);
    append_property<nest::double_t>(d, "p_i", pi_);
    append_property<nest::double_t>(d, "p_j", pj_);
    append_property<nest::double_t>(d, "p_ij", pij_);

  }

} // of namespace nest
