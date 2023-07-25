from visdom import Visdom
import os

def visdom_initialization_sup(env, port):
    visdom = Visdom(env=env, port=port)
    visdom.line([0.], [0.], win='train_loss', opts=dict(title='Train Loss', xlabel='Epoch', ylabel='Train Loss', legend=['Train Loss'], width=550, height=350))
    visdom.line([0.], [0.], win='train_jc', opts=dict(title='Train Jc', xlabel='Epoch', ylabel='Train Jc', legend=['Train Jc'], width=550, height=350))
    visdom.line([0.], [0.], win='val_loss', opts=dict(title='Val Loss', xlabel='Epoch', ylabel='Val Loss', legend=['Val Loss'], width=550, height=350))
    visdom.line([0.], [0.], win='val_jc', opts=dict(title='Val Jc', xlabel='Epoch', ylabel='Val Jc', legend=['Val Jc'], width=550, height=350))
    return visdom

def visualization_sup(vis, epoch, train_loss, train_m_jc, val_loss, val_m_jc):
    vis.line([train_loss], [epoch], win='train_loss', update='append')
    vis.line([train_m_jc], [epoch], win='train_jc', update='append')
    vis.line([val_loss], [epoch], win='val_loss', update='append')
    vis.line([val_m_jc], [epoch], win='val_jc', update='append')

def visual_image_sup(vis, mask_train, pred_train, mask_val, pred_val):

    vis.heatmap(mask_train, win='train_mask', opts=dict(title='Train Mask', colormap='Viridis'))
    vis.heatmap(pred_train, win='train_pred1', opts=dict(title='Train Pred', colormap='Viridis'))
    vis.heatmap(mask_val, win='val_mask', opts=dict(title='Val Mask', colormap='Viridis'))
    vis.heatmap(pred_val, win='val_pred1', opts=dict(title='Val Pred', colormap='Viridis'))


def visdom_initialization_XNet(env, port):
    visdom = Visdom(env=env, port=port)
    visdom.line([[0., 0., 0., 0.]], [0.], win='train_loss', opts=dict(title='Train Loss', xlabel='Epoch', ylabel='Train Loss', legend=['Train Loss', 'Train Sup1', 'Train Sup2', 'Train Unsup'], width=550, height=350))
    visdom.line([[0., 0.]], [0.], win='train_jc', opts=dict(title='Train Jc', xlabel='Epoch', ylabel='Train Jc', legend=['Train Jc1', 'Train Jc2'], width=550, height=350))
    visdom.line([[0., 0.]], [0.], win='val_loss', opts=dict(title='Val Loss', xlabel='Epoch', ylabel='Val Loss', legend=['Val Sup1', 'Val Sup2'], width=550, height=350))
    visdom.line([[0., 0.]], [0.], win='val_jc', opts=dict(title='Val Jc', xlabel='Epoch', ylabel='Val Jc', legend=['Val Jc1', 'Val Jc2'], width=550, height=350))
    return visdom

def visualization_XNet(vis, epoch, train_loss, train_loss_sup1, train_loss_sup2, train_loss_cps, train_m_jc1, train_m_jc2, val_loss_sup1, val_loss_sup2, val_m_jc1, val_m_jc2):
    vis.line([[train_loss, train_loss_sup1, train_loss_sup2, train_loss_cps]], [epoch], win='train_loss', update='append')
    vis.line([[train_m_jc1, train_m_jc2]], [epoch], win='train_jc', update='append')
    vis.line([[val_loss_sup1, val_loss_sup2]], [epoch], win='val_loss', update='append')
    vis.line([[val_m_jc1, val_m_jc2]], [epoch], win='val_jc', update='append')

def visual_image_XNet(vis, mask_train, pred_train1, pred_train2, mask_val, pred_val1, pred_val2):

    vis.heatmap(mask_train, win='train_mask', opts=dict(title='Train Mask', colormap='Viridis'))
    vis.heatmap(pred_train1, win='train_pred1', opts=dict(title='Train Pred1', colormap='Viridis'))
    vis.heatmap(pred_train2, win='train_pred2', opts=dict(title='Train pred2', colormap='Viridis'))

    vis.heatmap(mask_val, win='val_mask', opts=dict(title='Val Mask', colormap='Viridis'))
    vis.heatmap(pred_val1, win='val_pred1', opts=dict(title='Val Pred1', colormap='Viridis'))
    vis.heatmap(pred_val2, win='val_pred2', opts=dict(title='Val Pred2', colormap='Viridis'))


def visdom_initialization_MT(env, port):
    visdom = Visdom(env=env, port=port)
    visdom.line([[0., 0., 0.]], [0.], win='train_loss', opts=dict(title='Train Loss', xlabel='Epoch', ylabel='Train Loss', legend=['Train Loss', 'Train Sup', 'Train Unsup'], width=550, height=350))
    visdom.line([0.], [0.], win='train_jc', opts=dict(title='Train Jc', xlabel='Epoch', ylabel='Train Jc', legend=['Train Jc'], width=550, height=350))
    visdom.line([[0., 0.]], [0.], win='val_loss', opts=dict(title='Val Loss', xlabel='Epoch', ylabel='Val Loss', legend=['Val Sup1', 'Val Sup2'], width=550, height=350))
    visdom.line([[0., 0.]], [0.], win='val_jc', opts=dict(title='Val Jc', xlabel='Epoch', ylabel='Val Jc', legend=['Val Jc1', 'Val Jc2'], width=550, height=350))
    return visdom

def visualization_MT(vis, epoch, train_loss, train_loss_sup1, train_loss_cps, train_m_jc1, val_loss_sup1, val_loss_sup2, val_m_jc1, val_m_jc2):
    vis.line([[train_loss, train_loss_sup1, train_loss_cps]], [epoch], win='train_loss', update='append')
    vis.line([train_m_jc1], [epoch], win='train_jc', update='append')
    vis.line([[val_loss_sup1, val_loss_sup2]], [epoch], win='val_loss', update='append')
    vis.line([[val_m_jc1, val_m_jc2]], [epoch], win='val_jc', update='append')

def visual_image_MT(vis, mask_train, pred_train1, mask_val, pred_val1, pred_val2):

    vis.heatmap(mask_train, win='train_mask', opts=dict(title='Train Mask', colormap='Viridis'))
    vis.heatmap(pred_train1, win='train_pred1', opts=dict(title='Train Pred', colormap='Viridis'))
    vis.heatmap(mask_val, win='val_mask', opts=dict(title='Val Mask', colormap='Viridis'))
    vis.heatmap(pred_val1, win='val_pred1', opts=dict(title='Val Pred1', colormap='Viridis'))
    vis.heatmap(pred_val2, win='val_pred2', opts=dict(title='Val Pred2', colormap='Viridis'))


def visdom_initialization_EM(env, port):
    visdom = Visdom(env=env, port=port)
    visdom.line([[0., 0., 0.]], [0.], win='train_loss', opts=dict(title='Train Loss', xlabel='Epoch', ylabel='Train Loss', legend=['Train Loss', 'Train Sup', 'Train Unsup'], width=550, height=350))
    visdom.line([0.], [0.], win='train_jc', opts=dict(title='Train Jc', xlabel='Epoch', ylabel='Train Jc', legend=['Train Jc'], width=550, height=350))
    visdom.line([0.], [0.], win='val_loss', opts=dict(title='Val Loss', xlabel='Epoch', ylabel='Val Loss', legend=['Val Sup'], width=550, height=350))
    visdom.line([0.], [0.], win='val_jc', opts=dict(title='Val Jc', xlabel='Epoch', ylabel='Val Jc', legend=['Val Jc'], width=550, height=350))
    return visdom

def visualization_EM(vis, epoch, train_loss, train_loss_sup1, train_loss_cps, train_m_jc1, val_loss_sup1, val_m_jc1):
    vis.line([[train_loss, train_loss_sup1, train_loss_cps]], [epoch], win='train_loss', update='append')
    vis.line([train_m_jc1], [epoch], win='train_jc', update='append')
    vis.line([val_loss_sup1], [epoch], win='val_loss', update='append')
    vis.line([val_m_jc1], [epoch], win='val_jc', update='append')


def visdom_initialization_ConResNet(env, port):
    visdom = Visdom(env=env, port=port)
    visdom.line([[0., 0., 0.]], [0.], win='train_loss', opts=dict(title='Train Loss', xlabel='Epoch', ylabel='Train Loss', legend=['Train Loss', 'Train Seg', 'Train Res'], width=550, height=350))
    visdom.line([0.], [0.], win='train_jc', opts=dict(title='Train Jc', xlabel='Epoch', ylabel='Train Jc', legend=['Train Jc'], width=550, height=350))
    visdom.line([[0., 0.]], [0.], win='val_loss', opts=dict(title='Val Loss', xlabel='Epoch', ylabel='Val Loss', legend=['Val Seg', 'Val Res'], width=550, height=350))
    visdom.line([0.], [0.], win='val_jc', opts=dict(title='Val Jc', xlabel='Epoch', ylabel='Val Jc', legend=['Val Jc'], width=550, height=350))
    return visdom

def visualization_ConResNet(vis, epoch, train_loss, train_loss_seg, train_loss_res, train_m_jc1, val_loss_seg, val_loss_res, val_m_jc1):
    vis.line([[train_loss, train_loss_seg, train_loss_res]], [epoch], win='train_loss', update='append')
    vis.line([train_m_jc1], [epoch], win='train_jc', update='append')
    vis.line([[val_loss_seg, val_loss_res]], [epoch], win='val_loss', update='append')
    vis.line([val_m_jc1], [epoch], win='val_jc', update='append')