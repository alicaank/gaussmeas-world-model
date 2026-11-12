"""Stage samplers with an explicit truncated last-step gradient contract."""
from contextlib import nullcontext
import torch


def autocast(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()


def schedule(steps, shift=5.):
    if steps<1 or shift<=0:raise ValueError('invalid solver schedule')
    t=torch.linspace(1,0,steps+1,dtype=torch.float64)
    return (shift*t/(1+(shift-1)*t)).tolist()


def velocity(model, condition, x, text, u, checkpoint=False, *, negative_text=None, guidance=1., model_kwargs=None):
    import math
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError('finite nonnegative guidance required')
    options = dict(model_kwargs or {})
    if set(options) - {'causal_context'}:
        raise ValueError('unsupported diagnostic forward override')
    with autocast(x.device):
        conditional = model(**condition.pack(x),text=text,u=u,checkpoint_blocks=checkpoint,**options).float()
        if guidance == 1.:return conditional
        if negative_text is None:raise ValueError('CFG requires explicit negative text embedding')
        unconditional = model(**condition.pack(x),text=negative_text,u=u,checkpoint_blocks=checkpoint,**options).float()
        return unconditional + guidance * (conditional-unconditional)


def last_step_input(model, condition, noise, text, steps, shift=5., **options):
    times=schedule(steps,shift);x=noise.detach().clone()
    with torch.no_grad():
        for u,v in zip(times[:-2],times[1:-1]):
            x=x+(v-u)*velocity(model,condition,x,text,u,**options)
    return x.detach(),times[-2],times[-1]


def exit_step_input(model, condition, noise, text, steps, exit_index, shift=5., **options):
    """No-grad prefix of the solver, followed by a chosen clean-prediction exit.

    The exit predicts x0, not x at the next nonzero solver time. Keeping one
    differentiable exit preserves bounded graph memory while covering earlier
    denoising steps. This is independent of the score model's noising time.
    """
    if not isinstance(exit_index,int) or not 0 <= exit_index < steps:
        raise ValueError('exit index outside sampler schedule')
    times=schedule(steps,shift);x=noise.detach().clone()
    with torch.no_grad():
        for i in range(exit_index):
            u,v=times[i:i+2]
            x=x+(v-u)*velocity(model,condition,x,text,u,**options)
    return x.detach(),times[exit_index],0.


def last_step(model, condition, cached, text, grad=True, **options):
    x,u,v=cached
    with torch.enable_grad() if grad else torch.no_grad():
        # Projections are recomputed here; no detached trainable condition cache.
        y=x+(v-u)*velocity(model,condition,x,text,u,checkpoint=grad,**options)
    if not torch.isfinite(y).all():raise RuntimeError('nonfinite sample')
    return y


def sample(model, condition, noise, text, steps=30, last_step_grad=False, **options):
    cached=last_step_input(model,condition,noise,text,steps,**options)
    return last_step(model,condition,cached,text,grad=last_step_grad,**options)


def sequential_pair_backward(model, conditions, noises, text, steps, observe, objective, weight=1.):
    """Nine forward calls at four steps, one live branch graph, exact pair gradient
    for the stated last-step surrogate. Model/RNG state is replayed per branch.
    observe(branch, latent) must preserve its latent-input graph.
    """
    inputs=[last_step_input(model,c,n,text,steps) for c,n in zip(conditions,noises)]
    def rng():return (torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    def restore(s):
        torch.set_rng_state(s[0])
        if s[1]:torch.cuda.set_rng_state_all(s[1])
    before=rng()
    with torch.no_grad():a=observe(0,last_step(model,conditions[0],inputs[0],text,False)).detach()
    after=rng()
    b=observe(1,last_step(model,conditions[1],inputs[1],text,True))
    loss=objective(a,b)*weight
    if not torch.isfinite(loss):raise RuntimeError('nonfinite paired loss')
    value=float(loss.detach()); b_saved=b.detach()
    loss.backward();del loss,b
    final_rng=rng();restore(before)
    a=observe(0,last_step(model,conditions[0],inputs[0],text,True))
    loss=objective(a,b_saved)*weight
    loss.backward();restore(final_rng)
    return value
