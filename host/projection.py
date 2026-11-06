from fractions import Fraction
import numpy as np


def integer_array(values, shape, label):
    array = np.asarray(values)
    if array.shape != shape or not np.issubdtype(array.dtype, np.integer):
        raise ValueError('Expected integer '+label)
    if np.any(array < -(1 << 31)) or np.any(array > (1 << 31)-1):
        raise ValueError('Integer outside supported range: '+label)
    return array.astype(np.int64, copy=False)


def prepare_completion(metadata):
    permutation = integer_array(metadata['input_permutation'], (1024,), 'input permutation')
    features = metadata['shared_slice_features']
    if permutation.shape != (1024,) or sorted(permutation.tolist()) != list(range(1024)):
        raise ValueError('Input permutation must cover every model feature once')
    if not features or len(set(features)) != len(features) or any(
            not isinstance(c, int) or not 0 <= c < 1024 for c in features):
        raise ValueError('Invalid shared slice features')
    ids = metadata['token_ids']
    if len(ids) != 4 or len(set(ids)) != 4 or metadata['full_token_id'] not in ids:
        raise ValueError('Expected four distinct output neurons and one complete neuron')
    full_row = ids.index(metadata['full_token_id'])
    original = integer_array(metadata['full_weights_model_order'], (4, 1024), 'full weights')
    resident = integer_array(metadata['weights'], (4, 1024), 'resident weights')
    remaining = integer_array(metadata['host_remaining_weights'], (4, 1024), 'host weights')
    if np.any(original < -128) or np.any(original > 127):
        raise ValueError('Weight outside signed INT8 range')
    owned = np.tile(np.isin(permutation, features), (4, 1))
    owned[full_row] = True
    full_wire = original[:, permutation]
    if not np.array_equal(resident, np.where(owned, full_wire, 0)):
        raise ValueError('Resident coefficients do not match their declared ownership')
    if not np.array_equal(remaining[:, permutation], np.where(owned, 0, full_wire)):
        raise ValueError('Host coefficients overlap or omit resident coefficients')
    if metadata['resident_feature_counts'] != owned.sum(axis=1).tolist():
        raise ValueError('Resident feature counts do not match the ownership mask')
    terms = []
    for row in range(4):
        columns = np.flatnonzero(~owned[row])
        if len(columns):
            terms.append((row, columns, full_wire[row, columns].copy()))
    return dict(host_terms=terms, full_row=full_row, permutation=permutation,
                chip_abs_bounds=(128*np.abs(resident).sum(axis=1)).tolist(),
                host_coefficient_count=sum(len(columns) for _, columns, _ in terms))


def complete_dot_products(chip_dots, wire_activations, plan):
    values = integer_array(wire_activations, (1024,), 'activations')
    if np.any(values < -128) or np.any(values > 127):
        raise ValueError('Activation outside signed INT8 range')
    result = integer_array(chip_dots, (4,), 'chip sums').tolist()
    if any(abs(v) > bound for v, bound in zip(result, plan['chip_abs_bounds'])):
        raise ValueError('Chip sum exceeds the resident coefficient bound')
    for row, columns, weights in plan['host_terms']:
        result[row] += int(weights @ values[columns])
    return result


def select_completed(dots, metadata):
    dots = integer_array(dots, (4,), 'completed dot products')
    scores = [int(dot)*scale for dot, scale in zip(dots, metadata['scale_numerators'])]
    row = max(range(4), key=lambda i: (scores[i], -metadata['token_ids'][i]))
    return metadata['token_ids'][row], Fraction(scores[row], metadata['scale_denominator'])


def classify_winner(token, metadata):
    if token == metadata['full_token_id']:
        return 'complete_chip_neuron'
    if token in metadata['token_ids']:
        return 'chip_slice_plus_host_complement'
    return 'host_nonresident_neuron'
