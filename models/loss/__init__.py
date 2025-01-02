from models.loss.adv_loss import adv
from models.loss.coral import CORAL
from models.loss.cos import cosine
from models.loss.kl_js import kl_div, js
from models.loss.mmd import MMD_loss
from models.loss.mutual_info import Mine
from models.loss.pair_dist import pairwise_dist

__all__ = [
    'adv',
    'CORAL',
    'cosine',
    'kl_div',
    'js'
    'MMD_loss',
    'Mine',
    'pairwise_dist'
]