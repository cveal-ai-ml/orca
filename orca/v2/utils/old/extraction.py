"""
Author: Charlie
Purpose: Extraction tools
"""


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def get_tsne_features(data, num_features):

    # data = data.samples.reshape(data.samples.shape[0], -1)

    model = TSNE(n_components=num_features)

    return model.fit_transform(data)


def get_pca_features(data, num_features):

    # data = data.samples.reshape(data.samples.shape[0], -1)

    model = PCA(n_components=num_features)

    return model.fit_transform(data)


def get_embeddings(predictions, num_features=2):

    pca = get_pca_features(predictions, num_features)
    tsne = get_tsne_features(predictions, num_features)

    return tsne, pca
