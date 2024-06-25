import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    feature_vector = np.asarray(feature_vector)
    target_vector = np.asarray(target_vector)
    target_vector = target_vector[np.argsort(feature_vector)]
    feature_vector = np.sort(feature_vector)
    left_prob = np.cumsum(target_vector)[:len(target_vector) - 1] / np.arange(1, len(target_vector))
    right_prob = np.cumsum(target_vector[::-1])[:-1][::-1] / np.arange(len(target_vector) - 1, 0, -1)
    ginis_left = 1 - left_prob ** 2 - (1 - left_prob) ** 2
    ginis_right = 1 - right_prob ** 2 - (1 - right_prob) ** 2
    ginis = -1 * np.arange(1, len(target_vector)) / len(target_vector) * ginis_left - (
                len(target_vector) - np.arange(1, len(target_vector))) / len(target_vector) * ginis_right
    val, ind, cnt = np.unique(feature_vector, return_index=True, return_counts=True)
    thresholds = (val[1:] + val[:len(val) - 1]) / 2
    ginis = ginis[(cnt - 1 + ind)[:-1]]
    threshold_best = thresholds[np.argmax(ginis)]
    gini_best = np.max(ginis)
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._depth = 0

    def _check(self, feature_vector, threshold):
        split = feature_vector < threshold
        return np.sum(split) >= self._min_samples_leaf and np.sum(~split) >= self._min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        self._depth += 1
        if np.all(sub_y == sub_y[0]):  # should be == instead !=
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            self._depth -= 1
            return

        elif (self._max_depth and self._depth == self._max_depth) or (self._min_samples_split and sub_X.shape[
            0] < self._min_samples_split):  # checking max depth and sample size
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]  # return list of tuples
            self._depth -= 1
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):  # from 0 to X.shape[1]
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count  # click < count that's why click is divided by count
                sorted_categories = list(
                    map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))  # contains categories
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))  # list is missed
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1:  # if all values is equal that this feature is suck
                continue

            ths, gns, threshold, gini = find_best_split(feature_vector, sub_y)
            if self._min_samples_leaf:
                split_ = feature_vector < threshold
                if np.sum(split_) < self._min_samples_leaf or np.sum(~split_) < self._min_samples_leaf:
                    mask = np.asarray(list(map(self._check, feature_vector, ths)))
                    if not np.sum(mask):
                        continue
                    threshold = ths[mask][np.argmax(gns[mask])]
                    gini = np.max(gns[mask])

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":  # lower c
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]  # return list of tuples
            self._depth -= 1
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"])  # should be ~ or logical not
        self._depth -= 1

    def _predict_node(self, x, node):
        if node["type"] == 'terminal':
            return node["class"]
        else:
            feature = node["feature_split"]
            if self._feature_types[feature] == "real":
                if x[feature] < node["threshold"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])
            elif self._feature_types[feature] == "categorical":
                if x[feature] in node["categories_split"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])
            else:
                raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_depth_of_node(self, node):
        if node['type'] == 'terminal':
            return 0
        return max(self.get_depth_of_node(node['left_child']), self.get_depth_of_node(node['right_child'])) + 1

    def get_depth(self):
        return self.get_depth_of_node(self._tree) + 1

    def get_params(self, deep=False):
        return {'feature_types': self._feature_types,
                'max_depth': self._max_depth,
                'min_samples_split': self._min_samples_split,
                'min_samples_leaf': self._min_samples_leaf}
