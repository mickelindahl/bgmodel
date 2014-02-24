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

  BCPNNDopaConnection::BCPNNDopaConnection() :
    nest::ConnectionHetWD(),
    yi_(0.0),             /* initial conditions */
    yj_(0.0),
    taui_(10.0),
    tauj_(10.0),
    taue_(100.0),
    taup_(1000.0),
    epsilon_(0.001),
    fmax_(50.0),
    K_(1.0),
    gain_(1.0),
    zi_(0.01), 		
    zj_(0.01),
    ei_(0.01),
    ej_(0.01),
    eij_(0.001),
    pi_(0.01),
    pj_(0.01),
    pij_(0.001),
    bias_(0.0),

    b_(0.0),
    Kmin_(0.0),
    Kmax_(200.0),
    reverse_(1.0),
    tau_c_(1000.0),
    tau_n_(200.0),
    vt_(0),

    dopamine_modulated_(false)




  { }

  BCPNNDopaConnection::BCPNNDopaConnection(const BCPNNDopaConnection &rhs) :
    nest::ConnectionHetWD(rhs)
  {
    yi_ = rhs.yi_;
    yj_ = rhs.yj_;
    taui_ = rhs.taui_;
    tauj_ = rhs.tauj_;
    taue_ = rhs.taue_;
    taup_ = rhs.taup_;
    epsilon_ = rhs.epsilon_;
	gain_ = rhs.gain_;
    fmax_ = rhs.fmax_;
    K_ = rhs.K_;
    zi_ = rhs.zi_;
    zj_ = rhs.zj_;
    ei_ = rhs.ei_;
    ej_ = rhs.ej_;
    eij_ = rhs.eij_;
    pi_ = rhs.pi_;
    pj_ = rhs.pj_;
    pij_ = rhs.pij_;
    bias_ = rhs.bias_;


    b_=rhs.b_;
    Kmin_=rhs.Kmin_;
    Kmax_=rhs.Kmax_;
    reverse_=rhs.reverse_;
    tau_c_=rhs.tau_c_;
    tau_n_=rhs.tau_n_;
    vt_=rhs.vt_;

    dopamine_modulated_=rhs.dopamine_modulated_;
  }

  void BCPNNDopaConnection::get_status(DictionaryDatum & d) const
  {
    nest::ConnectionHetWD::get_status(d);
    def<nest::double_t>(d, "tau_i", taui_);
    def<nest::double_t>(d, "tau_j", tauj_);
    def<nest::double_t>(d, "tau_e", taue_);
    def<nest::double_t>(d, "tau_p", taup_);
    def<nest::double_t>(d, "epsilon", epsilon_);
    def<nest::double_t>(d, "fmax", fmax_);
    def<nest::double_t>(d, "bias", bias_);
    def<nest::double_t>(d, "K", K_);
    def<nest::double_t>(d, "gain", gain_);
    def<nest::double_t>(d, "p_i", pi_);
    def<nest::double_t>(d, "p_j", pj_);
    def<nest::double_t>(d, "p_ij", pij_);

    def<nest::double_t>(d, "b", b_);
    def<nest::double_t>(d, "Kmin", Kmin_);
    def<nest::double_t>(d, "Kmax", Kmax_);
    def<nest::double_t>(d, "reverse", reverse_);
    def<nest::double_t>(d, "tau_c", tau_c_);
    def<nest::double_t>(d, "tau_n", tau_n_);

    def<bool>(d, "dopamine_modulated", dopamine_modulated_);
  }

  void BCPNNDopaConnection::set_status(const DictionaryDatum & d, nest::ConnectorModel &cm)
  {
    nest::ConnectionHetWD::set_status(d, cm);
    updateValue<nest::double_t>(d, "tau_i", taui_);
    updateValue<nest::double_t>(d, "tau_j", tauj_);
    updateValue<nest::double_t>(d, "tau_e", taue_);
    updateValue<nest::double_t>(d, "tau_p", taup_);
    updateValue<nest::double_t>(d, "K", K_);
    updateValue<nest::double_t>(d, "epsilon", epsilon_);
    updateValue<nest::double_t>(d, "fmax", fmax_);
    updateValue<nest::double_t>(d, "bias", bias_);
    updateValue<nest::double_t>(d, "gain", gain_);
    updateValue<nest::double_t>(d, "p_i", pi_);
    updateValue<nest::double_t>(d, "p_j", pj_);
    updateValue<nest::double_t>(d, "p_ij", pij_);


    updateValue<nest::double_t>(d, "b", b_);
    updateValue<nest::double_t>(d, "Kmin", Kmin_);
    updateValue<nest::double_t>(d, "Kmax", Kmax_);
    updateValue<nest::double_t>(d, "reverse", reverse_);
    updateValue<nest::double_t>(d, "tau_c", tau_c_);
    updateValue<nest::double_t>(d, "tau_n", tau_n_);

    updateValue<bool>(d, "dopamine_modulated", dopamine_modulated_);

  }

   /**
   * Set properties of this connection from position p in the properties
   * array given in dictionary.
   */
  void BCPNNDopaConnection::set_status(const DictionaryDatum & d, nest::index p, nest::ConnectorModel &cm)
  {
    nest::ConnectionHetWD::set_status(d, p, cm);
    nest::set_property<nest::double_t>(d, "tau_i", p, taui_);
    nest::set_property<nest::double_t>(d, "tau_j", p, tauj_);
    nest::set_property<nest::double_t>(d, "tau_e", p, taue_);
    nest::set_property<nest::double_t>(d, "tau_p", p, taup_);
    nest::set_property<nest::double_t>(d, "K", p, K_);
    nest::set_property<nest::double_t>(d, "epsilon", p, epsilon_);
    nest::set_property<nest::double_t>(d, "fmax", p, fmax_);
    nest::set_property<nest::double_t>(d, "bias", p, bias_);
    nest::set_property<nest::double_t>(d, "gain", p, gain_);
    nest::set_property<nest::double_t>(d, "p_i", p, pi_);
    nest::set_property<nest::double_t>(d, "p_j", p, pj_);
    nest::set_property<nest::double_t>(d, "p_ij", p, pij_);

    nest::set_property<nest::double_t>(d, "b",p, b_);
    nest::set_property<nest::double_t>(d, "Kmin", p,Kmin_);
    nest::set_property<nest::double_t>(d, "Kmax", p,Kmax_);
    nest::set_property<nest::double_t>(d, "reverse", p,reverse_);
    nest::set_property<nest::double_t>(d, "tau_c", p,tau_c_);
    nest::set_property<nest::double_t>(d, "tau_n", p,tau_n_);

    nest::set_property<bool>(d, "dopamine_modulated", p,dopamine_modulated_);
  }

  void BCPNNDopaConnection::initialize_property_arrays(DictionaryDatum & d) const
  {
    nest::ConnectionHetWD::initialize_property_arrays(d);
    initialize_property_array(d, "tau_i");
    initialize_property_array(d, "tau_j");
    initialize_property_array(d, "tau_e");
    initialize_property_array(d, "tau_p");
    initialize_property_array(d, "K");
    initialize_property_array(d, "epsilon");
    initialize_property_array(d, "fmax");
    initialize_property_array(d, "bias");
    initialize_property_array(d, "gain");
    initialize_property_array(d, "p_i");
    initialize_property_array(d, "p_j");
    initialize_property_array(d, "p_ij");


    initialize_property_array(d, "b");
    initialize_property_array(d, "Kmin");
    initialize_property_array(d, "Kmax");
    initialize_property_array(d, "reverse");
    initialize_property_array(d, "tau_c");
    initialize_property_array(d, "tau_n");

    initialize_property_array(d, "dopamine_modulated_");
  }

  /**
   * Append properties of this connection to the given dictionary. If the
   * dictionary is empty, new arrays are created first.
   */
  void BCPNNDopaConnection::append_properties(DictionaryDatum & d) const
  {
    nest::ConnectionHetWD::append_properties(d);
    append_property<nest::double_t>(d, "tau_i", taui_);
    append_property<nest::double_t>(d, "tau_j", tauj_);
    append_property<nest::double_t>(d, "tau_e", taue_);
    append_property<nest::double_t>(d, "tau_p", taup_);
    append_property<nest::double_t>(d, "K", K_);
    append_property<nest::double_t>(d, "epsilon", epsilon_);
    append_property<nest::double_t>(d, "fmax", fmax_);
    append_property<nest::double_t>(d, "bias", bias_);
    append_property<nest::double_t>(d, "gain", gain_);
    append_property<nest::double_t>(d, "p_i", pi_);
    append_property<nest::double_t>(d, "p_j", pj_);
    append_property<nest::double_t>(d, "p_ij", pij_);

    append_property<nest::double_t>(d, "b", b_);
    append_property<nest::double_t>(d, "Kmin", Kmin_);
    append_property<nest::double_t>(d, "Kmax", Kmax_);
    append_property<nest::double_t>(d, "reverse", reverse_);
    append_property<nest::double_t>(d, "tau_c", tau_c_);
    append_property<nest::double_t>(d, "tau_n", tau_n_);

    append_property<bool>(d, "dopamine_modulated", dopamine_modulated_);
  }

} // of namespace nest
