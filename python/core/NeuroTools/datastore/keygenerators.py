"""
Key generators for data store objects
"""
import hashlib
import pickle
import sys
import os.path


def full_type(component):
    """Returns a string representing the full type of the component."""
    if component.__class__.__name__ == 'module':  # component is a module
        if component.__name__ == "__main__":
            return os.path.basename(sys.argv[0][:-3])
        else:
            return component.__name__
    else:
        return component.__module__ + '.' + component.__class__.__name__


def hash_pickle(component):
    """
    Key generator.

    Use pickle to convert the component state dictionary to a string, then
    hash this string to give a unique identifier of fixed length.
    """
    state = {'type': full_type(component),
             'version': component.version,
             #'parameters_uri': component.parameters._url}
             'parameters': component.parameters}
    if component.input is None:
        state['input'] = 'None'
    else:
        state['input'] = hash_pickle(component.input)
    return hashlib.sha1(pickle.dumps(state)).hexdigest()


def join_with_underscores(component):
    """
    Key generator.

    Return a string that contains all necessary information about the
    component state.
    """
    s = "%s-r%s_%s" % (full_type(component),
                       component.version,
                       #component.parameters._url)
                       component.parameters)
    if component.input is not None:
        s += "%s" % join_with_underscores(component.input)
    # remove characters that don't go well in filesystem paths
    replace = lambda s, r: s.replace(r[0], r[1])
    replacements = [('/', '_'), (' ', '_'), ('[', ''),
                    (']', ''), (':', ''), (',', '')]
    s = reduce(replace, [s] + replacements)
    return s
