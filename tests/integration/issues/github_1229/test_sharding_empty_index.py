import os

import numpy as np

from jina import Flow, Document

PORT = 45888


def test_sharding_empty_index(random_workspace_name):
    print(f'WORKSPACE = {random_workspace_name}')
    os.environ['JINA_TEST_1229_WORKSPACE'] = os.path.abspath(random_workspace_name)
    num_shards = 2

    f = Flow() \
        .add(
        uses='vectorindexer.yml',
        shards=num_shards,
        separated_workspace=True,
        uses_after='_merge_all'
    )

    num_docs = 1
    data = []
    for i in range(num_docs):
        doc = Document()
        doc.content = f'data {i}'
        doc.embedding = np.array([i])
        doc.update_id()
        data.append(doc)

    with f:
        f.index(data)

    num_query = 10
    query = []
    for i in range(num_query):
        doc = Document()
        doc.content = f'query {i}'
        doc.embedding = np.array([i])
        doc.update_id()
        query.append(doc)

    def callback(result):
        for d in result.docs:
            print(f'document: {d.content}')
            print(f'matches = {[m.content for m in d.matches]}')
        assert result is not None

    with f:
        f.search(query, output_fn=callback)
