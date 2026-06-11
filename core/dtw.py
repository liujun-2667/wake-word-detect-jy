import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def cosine_distance(x, y):
    dot = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        return 1.0
    return 1.0 - dot / (norm_x * norm_y)


def dtw_distance(sequence1, sequence2, distance_func=None, window=None):
    if distance_func is None:
        distance_func = euclidean_distance

    n = len(sequence1)
    m = len(sequence2)

    if window is None:
        window = max(n, m)

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        start = max(1, i - window)
        end = min(m + 1, i + window + 1)
        for j in range(start, end):
            cost = distance_func(sequence1[i - 1], sequence2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )

    distance = dtw_matrix[n, m]
    return distance, dtw_matrix


def dtw_path(dtw_matrix):
    n, m = dtw_matrix.shape
    n -= 1
    m -= 1

    path = []
    i, j = n, m

    while i > 0 or j > 0:
        path.append((i - 1, j - 1))

        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_val = min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )
            if min_val == dtw_matrix[i - 1, j - 1]:
                i -= 1
                j -= 1
            elif min_val == dtw_matrix[i - 1, j]:
                i -= 1
            else:
                j -= 1

    path.append((0, 0))
    path.reverse()
    return path


def dtw_similarity(sequence1, sequence2, distance_func=None, window=None):
    distance, _ = dtw_distance(sequence1, sequence2, distance_func, window)
    max_len = max(len(sequence1), len(sequence2))
    normalized_distance = distance / max_len
    similarity = 1.0 / (1.0 + normalized_distance)
    return similarity


def multi_template_similarity(sequence, templates, distance_func=None, window=None):
    similarities = []
    for template in templates:
        sim = dtw_similarity(sequence, template, distance_func, window)
        similarities.append(sim)
    return max(similarities) if similarities else 0.0


def average_template(templates):
    if not templates:
        return None

    ref_template = templates[0]
    aligned_templates = [ref_template]

    for template in templates[1:]:
        _, dtw_matrix = dtw_distance(ref_template, template)
        path = dtw_path(dtw_matrix)

        aligned = np.zeros_like(ref_template)
        counts = np.zeros(len(ref_template))

        for i, j in path:
            if i < len(ref_template) and j < len(template):
                aligned[i] += template[j]
                counts[i] += 1

        counts = np.maximum(counts, 1)
        aligned = aligned / counts[:, np.newaxis]
        aligned_templates.append(aligned)

    avg_template = np.mean(aligned_templates, axis=0)
    return avg_template
