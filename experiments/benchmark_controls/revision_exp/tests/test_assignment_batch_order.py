from revision_exp.audit.e0 import batch_local_graph_dependency


def test_target_assignment_can_depend_on_cobatch_cells():
    evidence = batch_local_graph_dependency()
    assert evidence["assignment_changed"] is True
    assert evidence["target_similarity_max_abs_change"] > 0
