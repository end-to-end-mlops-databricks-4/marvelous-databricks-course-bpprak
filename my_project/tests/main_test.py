from my_project.main import get_spark, get_taxis


def test_main():
    taxis = get_taxis(get_spark())
    assert taxis.count() > 5
