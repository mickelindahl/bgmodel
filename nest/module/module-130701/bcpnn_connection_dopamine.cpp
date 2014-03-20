/*
 *  bcpnn_connection_dopamine.cpp
 *
 *  Written by Mikael Lindahl
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
   nest::CommonSynapseProperties(),

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
	nest::CommonSynapseProperties::get_status(d);
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
	ej_c_(0.01),
	eij_(0.0001),
	eij_c_(0.0001),
	k_(0.0),
	n_(0.0),
	n_add_(0.0),
	m_(0.0),
	pi_(0.01),
	pj_(0.01),
	pij_(0.0001),
	spike_idx_(0),
	t_last_update_(0.0),
	yi_(0.0),              //initial conditions
	yj_(0.0),
	zi_(0.01),
	zj_(0.01),
	zj_c_(0.01)
{}

  BCPNNDopaConnection::BCPNNDopaConnection(const BCPNNDopaConnection &rhs) :
    nest::ConnectionHetWD(rhs)
  {
	bias_ = rhs.bias_;
	BUFFER_ = rhs.BUFFER_;
    dopa_spikes_idx_ = rhs.dopa_spikes_idx_;
    ei_ = rhs.ei_;
    ej_ = rhs.ej_;
    ej_c_ = rhs.ej_c_;
    eij_ = rhs.eij_;
    eij_c_ = rhs.eij_c_;
    k_ = rhs.k_;
    n_ = rhs.n_;
    n_add_ = rhs.n_add_;
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
    zj_c_ = rhs.zj_c_;
  }

  void BCPNNDopaConnection::get_status(DictionaryDatum & d) const
  {
    nest::ConnectionHetWD::get_status(d);

	def<nest::double_t>(d, "bias", bias_);
    def<nest::double_t>(d, "e_i", ei_);
    def<nest::double_t>(d, "e_j", ej_);
    def<nest::double_t>(d, "e_j_c", ej_c_);
    def<nest::double_t>(d, "e_ij", eij_);
    def<nest::double_t>(d, "e_ij_c", eij_c_);
    def<nest::double_t>(d, "k", k_);
    def<nest::double_t>(d, "n", n_);
    def<nest::double_t>(d, "m", m_);
    def<nest::double_t>(d, "p_i", pi_);
    def<nest::double_t>(d, "p_j", pj_);
    def<nest::double_t>(d, "p_ij", pij_);
    def<nest::double_t>(d, "y_i", yi_);
    def<nest::double_t>(d, "y_j", yj_);
    def<nest::double_t>(d, "z_i", zj_);
    def<nest::double_t>(d, "z_j", zj_);
    def<nest::double_t>(d, "z_j_c", zj_c_);
  }

  void BCPNNDopaConnection::set_status(const DictionaryDatum & d,
		  nest::ConnectorModel &cm)
  {
	// base class properties
	nest::ConnectionHetWD::set_status(d, cm);

    updateValue<nest::double_t>(d, "bias", bias_);
    updateValue<nest::double_t>(d, "e_i", ei_);
    updateValue<nest::double_t>(d, "e_j", ej_);
    updateValue<nest::double_t>(d, "e_j_c", ej_c_);
    updateValue<nest::double_t>(d, "e_ij", eij_);
    updateValue<nest::double_t>(d, "e_ij_c", eij_c_);
    updateValue<nest::double_t>(d, "k", k_);
    updateValue<nest::double_t>(d, "n", n_);
    updateValue<nest::double_t>(d, "m", m_);
    updateValue<nest::double_t>(d, "p_i", pi_);
    updateValue<nest::double_t>(d, "p_j", pj_);
    updateValue<nest::double_t>(d, "p_ij", pij_);
    updateValue<nest::double_t>(d, "y_i", yi_);
    updateValue<nest::double_t>(d, "y_j", yj_);
    updateValue<nest::double_t>(d, "z_i", zi_);
    updateValue<nest::double_t>(d, "z_j", zj_);
    updateValue<nest::double_t>(d, "z_j_c", zj_c_);
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
    nest::set_property<nest::double_t>(d, "e_i", p, ei_);
    nest::set_property<nest::double_t>(d, "e_j", p, ej_);
    nest::set_property<nest::double_t>(d, "e_j_c", p, ej_c_);
     nest::set_property<nest::double_t>(d, "e_ij", p, eij_);
    nest::set_property<nest::double_t>(d, "e_ij_c", p, eij_c_);
    nest::set_property<nest::double_t>(d, "k", p, k_);
    nest::set_property<nest::double_t>(d, "n", p, n_);
    nest::set_property<nest::double_t>(d, "m", p, m_);
    nest::set_property<nest::double_t>(d, "p_i", p, pi_);
    nest::set_property<nest::double_t>(d, "p_j", p, pj_);
    nest::set_property<nest::double_t>(d, "p_ij", p, pij_);
    nest::set_property<nest::double_t>(d, "y_i", p, yi_);
    nest::set_property<nest::double_t>(d, "y_j", p, yj_);
    nest::set_property<nest::double_t>(d, "z_i", p, zi_);
    nest::set_property<nest::double_t>(d, "z_j", p, zj_);
    nest::set_property<nest::double_t>(d, "z_j_c", p, zj_c_);
  }

  void BCPNNDopaConnection::initialize_property_arrays(DictionaryDatum & d) const
  {
    nest::ConnectionHetWD::initialize_property_arrays(d);

    initialize_property_array(d, "bias");
    initialize_property_array(d, "e_i");
    initialize_property_array(d, "e_j");
    initialize_property_array(d, "e_j_c");
    initialize_property_array(d, "e_ij");
    initialize_property_array(d, "e_ij_c");
    initialize_property_array(d, "k");
    initialize_property_array(d, "n");
    initialize_property_array(d, "m");
    initialize_property_array(d, "p_i");
    initialize_property_array(d, "p_j");
    initialize_property_array(d, "p_ij");
    initialize_property_array(d, "y_i");
    initialize_property_array(d, "y_j");
    initialize_property_array(d, "z_i");
    initialize_property_array(d, "z_j");
    initialize_property_array(d, "z_j_c");
  }

  /**
   * Append properties of this connection to the given dictionary. If the
   * dictionary is empty, new arrays are created first.
   */
  void BCPNNDopaConnection::append_properties(DictionaryDatum & d) const
  {
    nest::ConnectionHetWD::append_properties(d);

    append_property<nest::double_t>(d, "bias", bias_);
    append_property<nest::double_t>(d, "e_i", ei_);
    append_property<nest::double_t>(d, "e_j", ej_);
    append_property<nest::double_t>(d, "e_j_c", ej_c_);
    append_property<nest::double_t>(d, "e_ij", eij_);
    append_property<nest::double_t>(d, "e_ij_C", eij_c_);
    append_property<nest::double_t>(d, "k", k_);
    append_property<nest::double_t>(d, "n", n_);
    append_property<nest::double_t>(d, "m", m_);
    append_property<nest::double_t>(d, "p_i", pi_);
    append_property<nest::double_t>(d, "p_j", pj_);
    append_property<nest::double_t>(d, "p_ij", pij_);
    append_property<nest::double_t>(d, "y_i", yi_);
    append_property<nest::double_t>(d, "y_j", yj_);
    append_property<nest::double_t>(d, "z_i", zi_);
    append_property<nest::double_t>(d, "z_j", zj_);
    append_property<nest::double_t>(d, "z_j_c", zj_c_);
  }

} // of namespace nest
