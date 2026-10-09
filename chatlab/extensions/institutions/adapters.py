"""The two supported scenarios. Invoice v1 remains its existing compatibility adapter."""
from . import support


def adapter(scenario):
    if scenario == 'customer_support':
        return support
    if scenario == 'invoice_payments':
        from . import bundles
        return bundles
    raise ValueError(f'Unsupported Institutions scenario: {scenario!r}')
