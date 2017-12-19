# Create by Mikael Lindahl on 12/2/17.

import json
import pprint
import nest

pp = pprint.pprint


def extract_nodes(nodes, file_path):
    params = {}

    for key, pop in nodes.items():

        status = nest.GetStatus(pop)
        V_m = sum([s['V_m'] for s in status]) / len(pop)
        V_th = sum([s['V_th'] for s in status]) / len(pop)
        C_m = sum([s['C_m'] for s in status]) / len(pop)

        for s in status:
            del s['model']
            del s['recordables']
            del s['element_type']
            del s['thread_local_id']
            del s['thread']
            del s['archiver_length']
            del s['global_id']
            del s['tau_minus_triplet']
            del s['tau_minus']
            del s['tau_Ca']
            del s['t_spike']
            # del s['t_ref']
            del s['synaptic_elements']
            del s['vp']
            del s['supports_precise_spikes']
            del s['parent']
            del s['node_uses_wfr']
            del s['local_id']
            del s['local']
            # del s['gsl_error_tol']

        params[key] = {
            'nest': status[0],
            'size': len(pop),
            'rand': {
                'V_th': V_th,
                'V_m': V_m,
                'C_m': C_m
            }
        }

    f = open(file_path, 'w')

    json.dump(params, f)

    return params


def extract_connections(nodes, file_path):
    conns = {}
    for key1, pop1 in nodes.items():
        for key2, pop2 in nodes.items():
            #
            # if not (key1 == 'FS' and key2 == 'FS'):
            #     continue

            target = nest.GetStatus([pop2[0]])[0]

            trans = {}
            for k, v in target['receptor_types'].items():
                trans[v] = k

            key = key1 + '-' + key2

            conn = nest.GetConnections(nodes[key1], nodes[key2])

            # conns[key] =
            # pp(conn[0])

            status = nest.GetStatus(conn)
            # pp(status)

            # raise

            if not status:
                continue

            # print(status[0])

            dic_status = {}
            for s in status:

                del s['synapse_model']
                del s['target']
                del s['source']
                del s['sizeof']
                # del s['weight']

                if trans.get(s['receptor']):
                    skey = key + '-' + trans.get(s['receptor'])
                    s['receptor_type'] = trans.get(s['receptor'])
                else:
                    skey = key + '-' + str(s['receptor'])

                if dic_status.get(key):
                    dic_status[skey].append(s)

                else:
                    dic_status[skey] = [s]

            for skey, status in dic_status.items():

                if not status[0].get('weight'):
                    continue

                weight = sum([s['weight'] for s in status]) / len(status)

                conns[skey] = {
                    'total': len(status),
                    'fan_in': len(status) / float(len(nodes[key2])),
                    'rand': {
                        'weight': weight
                    },
                    'nest': status[0]
                }


                # pp(conns)
                # raise

    f = open(file_path, 'w')

    json.dump(conns, f)

    return conns
