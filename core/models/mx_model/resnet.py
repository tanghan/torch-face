"""Resnet."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mxnet as mx
from mxnet.gluon import nn
from gluon_face.single_task.recognition2.op.basic_layers import BatchNorm
from gluon_face.common.model_zoo.base_model import BaseModel
from gluon_face.common.model_zoo.se import SE
from gluon_face.common.model_zoo.eca import ECA
from gluon_face.common.model_zoo.cbam import CBAM
from gluon_face.single_task.recognition2.op.basic_layers import ConvOp


class EmbeddingUnit(mx.gluon.HybridBlock):
    def __init__(self, emb_type="E", emb_size=256):
        super(EmbeddingUnit, self).__init__()
        self.emb = nn.HybridSequential()
        if emb_type == "E":
            self.emb.add(BatchNorm())
            self.emb.add(nn.Dropout(0.4))
            self.emb.add(nn.Dense(emb_size))
            # fix_gamma
            self.emb.add(BatchNorm(scale=False))
        else:
            raise NotImplementedError

    def hybrid_forward(self, F, x):
        return self.emb(x)


class ResidualUnitV3(mx.gluon.HybridBlock):
    """Return ResNet_v3 Unit symbol for building ResNet.

    Refer to insightface residual_unit_v3,
    https://github.com/deepinsight/insightface/blob/master/src/
    symbols/fresnet.py

    Exclusively for face recognition :)

    Parameters
    ----------
    channels : int
        Number of output channels.
    strides : tuple
        Strides used in convolution.
    dim_match : boolean
        True means channel number between input and output is the same,
        otherwise means differ.
    bottle_neck : boolean
        True means bottleneck is used.
    groups : int
        Number of groups
    use_se : boolean
        True means Squeeze-and-Excitation Networks is used.

    """

    def __init__(self, channels, strides, dim_match, bottle_neck,
                 act_type='relu', groups=1, use_se=False,
                 use_cbam=False, use_eca=False, use_bias=False,
                 force_mirroring=False, workspace=1024, cudnn_tune=None):
        super(ResidualUnitV3, self).__init__()
        self.dim_match = dim_match
        self.use_se = use_se
        self.use_cbam = use_cbam
        self.use_eca = use_eca
        assert (use_se + use_cbam + use_eca) < 2, 'Choose 1 attention block at most!'  # noqa
        with self.name_scope():
            self.body_conv = nn.HybridSequential()
            if bottle_neck:
                ratio = 0.25
                if groups == 32:
                    ratio = 0.5
                elif groups == 64:
                    ratio = 1.0
                else:
                    assert groups == 1
                self.body_conv.add(BatchNorm())
                self.body_conv.add(ConvOp(channels=int(channels * ratio),
                                          kernel_size=1,
                                          op_list=('conv', 'bn', act_type),
                                          use_bias=use_bias,
                                          force_mirroring=force_mirroring,
                                          workspace=workspace,
                                          cudnn_tune=cudnn_tune))
                self.body_conv.add(ConvOp(channels=int(channels * ratio),
                                          kernel_size=3, strides=strides,
                                          padding=1,
                                          groups=groups,
                                          op_list=('conv', 'bn', act_type),
                                          use_bias=use_bias,
                                          force_mirroring=force_mirroring,
                                          workspace=workspace,
                                          cudnn_tune=cudnn_tune))
                self.body_conv.add(ConvOp(channels=channels,
                                          kernel_size=1,
                                          op_list=('conv', 'bn'),
                                          use_bias=use_bias,
                                          force_mirroring=force_mirroring,
                                          workspace=workspace,
                                          cudnn_tune=cudnn_tune))
                if not self.dim_match:
                    self.sc_conv = ConvOp(channels=channels,
                                          kernel_size=1,
                                          strides=strides,
                                          op_list=('conv', 'bn'),
                                          use_bias=use_bias,
                                          force_mirroring=force_mirroring,
                                          workspace=workspace,
                                          cudnn_tune=cudnn_tune)
            else:
                assert groups == 1
                self.body_conv.add(BatchNorm())
                self.body_conv.add(ConvOp(channels=channels,
                                          kernel_size=3,
                                          strides=1,
                                          padding=1,
                                          op_list=('conv', 'bn', act_type),
                                          use_bias=use_bias,
                                          force_mirroring=force_mirroring,
                                          workspace=workspace,
                                          cudnn_tune=cudnn_tune))
                self.body_conv.add(ConvOp(channels=channels,
                                          kernel_size=3,
                                          strides=strides,
                                          padding=1,
                                          op_list=('conv', 'bn'),
                                          use_bias=use_bias,
                                          force_mirroring=force_mirroring,
                                          workspace=workspace,
                                          cudnn_tune=cudnn_tune))
                if not self.dim_match:
                    self.sc_conv = ConvOp(channels=channels,
                                          kernel_size=3 if strides == 2 else 1,
                                          strides=strides,
                                          padding=1 if strides == 2 else 0,
                                          op_list=('conv', 'bn'),
                                          use_bias=use_bias,
                                          force_mirroring=force_mirroring,
                                          workspace=workspace,
                                          cudnn_tune=cudnn_tune)
            if use_se:
                self.attention = SE(units=channels)
            if use_cbam:
                self.attention = CBAM(units=channels)
            if use_eca:
                self.attention = ECA(units=channels)

    def hybrid_forward(self, F, x):
        if self.dim_match:
            shortcut = x
        else:
            shortcut = self.sc_conv(x)
        body = self.body_conv(x)
        if self.use_se or self.use_cbam or self.use_eca:
            body = self.attention(body)
        out = shortcut + body
        return out


class Resnet(BaseModel):
    """Return ResNet Unit block for building ResNet.

    Parameters
    ----------
    units : list
        Number of units in each stage.
    filter_list : list
        Channel size of each stage.
    bottle_neck : boolean
        Whether to use bottleneck.
    groups : int
        Number of groups
    use_se : boolean
        True means Squeeze-and-Excitation Networks is used.
    use_bias : boolean
        True means apply bias to convolution layers
    unit_type : str
        Resnet type to choose.

    """

    def __init__(self, units, filter_list, bottle_neck, groups=1,
                 use_se=False, use_cbam=False, use_eca=False,
                 act_type='relu', use_bias=False, unit_type='resnet_v3',
                 force_mirroring=False, workspace=1024, cudnn_tune=None,
                 input_version=1, emb_size=256, emb_type="E", **kwargs):
        super(Resnet, self).__init__(**kwargs)

        def unit_func(**res_kwargs):
            if unit_type == 'resnet_v3':
                return ResidualUnitV3(groups=groups,
                                      use_se=use_se,
                                      use_cbam=use_cbam,
                                      use_eca=use_eca,
                                      use_bias=use_bias,
                                      force_mirroring=force_mirroring,
                                      workspace=workspace,
                                      cudnn_tune=cudnn_tune,
                                      act_type=act_type,
                                      **res_kwargs)
            else:
                raise ValueError

        if self.stage1 is not None:
            if input_version == 0:
                self.stage1.add(ConvOp(channels=filter_list[0],
                                       kernel_size=3,
                                       strides=1, padding=1,
                                       op_list=('conv', 'bn', act_type),
                                       use_bias=use_bias,
                                       force_mirroring=force_mirroring,
                                       workspace=workspace,
                                       cudnn_tune=cudnn_tune))
            elif input_version == 1:
                self.stage1.add(ConvOp(channels=filter_list[0],
                                       kernel_size=3,
                                       strides=1, padding=1,
                                       op_list=('conv', 'bn', act_type),
                                       use_bias=use_bias,
                                       force_mirroring=force_mirroring,
                                       workspace=workspace,
                                       cudnn_tune=cudnn_tune))
                self.stage1.add(ConvOp(channels=filter_list[0],
                                       kernel_size=3,
                                       strides=1, padding=1,
                                       op_list=('conv', 'bn', act_type),
                                       use_bias=use_bias,
                                       force_mirroring=force_mirroring,
                                       workspace=workspace,
                                       cudnn_tune=cudnn_tune))
            else:
                raise ValueError

        if self.stage2 is not None:
            self.stage2.add(unit_func(channels=filter_list[1],
                                      strides=2,
                                      dim_match=False,
                                      bottle_neck=bottle_neck))
            for i in range(2, units[0] + 1):
                self.stage2.add(unit_func(channels=filter_list[1],
                                          strides=1,
                                          dim_match=True,
                                          bottle_neck=bottle_neck))

        if self.stage3 is not None:
            self.stage3.add(unit_func(channels=filter_list[2],
                                      strides=2,
                                      dim_match=False,
                                      bottle_neck=bottle_neck))
            for i in range(2, units[1] + 1):
                self.stage3.add(unit_func(channels=filter_list[2],
                                          strides=1,
                                          dim_match=True,
                                          bottle_neck=bottle_neck))

        if self.stage4 is not None:
            self.stage4.add(unit_func(channels=filter_list[3],
                                      strides=2,
                                      dim_match=False,
                                      bottle_neck=bottle_neck))
            for i in range(2, units[2] + 1):
                self.stage4.add(unit_func(channels=filter_list[3],
                                          strides=1,
                                          dim_match=True,
                                          bottle_neck=bottle_neck))

        if self.stage5 is not None:
            self.stage5.add(unit_func(channels=filter_list[4],
                                      strides=2,
                                      dim_match=False,
                                      bottle_neck=bottle_neck))
            for i in range(2, units[3] + 1):
                self.stage5.add(unit_func(channels=filter_list[4],
                                          strides=1,
                                          dim_match=True,
                                          bottle_neck=bottle_neck))

        self.emb_block = EmbeddingUnit(emb_type=emb_type,
                                       emb_size=emb_size)

    def hybrid_forward(self, F, x):
        out = super(Resnet, self).hybrid_forward(F, x)
        out = self.emb_block(out[-1])
        return out
