"""Task-independent literature priors: bounded preferences, never synthetic measurements."""
import numpy as np


def validate_prior(reply, names, X, papers):
    belief = reply.get('belief', {})
    if not belief or set(belief) - set(names):
        raise ValueError('A prior must name supported task parameters.')
    for name, pair in belief.items():
        j = names.index(name)
        if (len(pair) != 2 or not np.isfinite(pair).all()
                or not X[:, j].min() <= pair[0] <= X[:, j].max() or not 0 < pair[1] <= 1):
            raise ValueError('Prior centres must be in range and fractional widths in (0, 1].')
    sources = {p['id']: p for p in papers}
    citations = reply.get('citations', [])
    covered = set()
    for citation in citations:
        paper = sources.get(citation.get('id'))
        if (not paper or not citation.get('quote')
                or citation['quote'] not in paper['abstract'] or not citation.get('transfer_limit')):
            raise ValueError('Prior evidence needs a retrieved quote and transfer limitation.')
        supported = set(citation.get('parameters', []))
        if supported - set(belief):
            raise ValueError('Citations must refer to parameters in the prior.')
        covered.update(supported)
    if covered != set(belief) or not reply.get('reason') or not reply.get('uncertainty'):
        raise ValueError('Every prior parameter needs cited support and explicit uncertainty.')
    return {k: reply[k] for k in ('belief', 'citations', 'reason', 'uncertainty')}


def prior_weights(X, names, belief, observations):
    """At most 4:1 preference initially; fades with measurements and never excludes a region."""
    logp = np.zeros(len(X))
    span = np.where(np.ptp(X, axis=0) > 0, np.ptp(X, axis=0), 1)
    for name, (centre, width) in belief.items():
        j = names.index(name)
        logp -= 0.5 * ((X[:, j] - centre) / (span[j] * max(width, 0.05)))**2
    weight = 0.25 + 0.75 * np.exp(logp - logp.max())
    return weight ** (4 / (4 + observations))
