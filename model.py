from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import numpy as np
from model.module_encoder import TfModel, AudioConfig1, AudioConfig2, AudioConfig3
from model.until_module import PreTrainedModel, LayerNorm
from model.until_module import getBinaryTensor, CTCModule, MLLinear, MLAttention
import warnings
from model.losses import *
from mymodel.attention_modules import *
from model.until_config import *
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class TACNetPreTrainedModel(PreTrainedModel, nn.Module):
    def __init__(self, audio_config, *inputs, **kwargs):
        # utilize bert config as base config
        super(TACNetPreTrainedModel, self).__init__(audio_config)

        self.audio_config1 = audio_config
        self.audio_config2 = audio_config
        self.audio_config3 = audio_config
        self.audio = None


    @classmethod
    def from_pretrained(cls, audio1_model_name, audio2_model_name, audio3_model_name,
                        state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
        # 获取 task_config（如果传了）
        task_config = kwargs.get("task_config", None)

        # 设定 local_rank
        if task_config is not None:
            if not hasattr(task_config, "local_rank") or task_config.local_rank == -1:
                setattr(task_config, "local_rank", 0)

        # 获取每个音频模型的 config
        audio_config1, _ = AudioConfig1.get_config(audio1_model_name, cache_dir, type_vocab_size,
                                                   state_dict=None, task_config=task_config)
        audio_config2, _ = AudioConfig2.get_config(audio2_model_name, cache_dir, type_vocab_size,
                                                   state_dict=None, task_config=task_config)
        audio_config3, _ = AudioConfig3.get_config(audio3_model_name, cache_dir, type_vocab_size,
                                                   state_dict=None, task_config=task_config)

        # 构建模型（task_config 会作为 kwargs 传入 __init__，已支持）
        model = cls(audio_config1, audio_config2, audio_config3, *inputs, **kwargs)

        # 加载预训练权重
        if state_dict is not None:
            if isinstance(state_dict, str):
                # 如果 state_dict 是路径，自动加载
                state_dict = torch.load(state_dict, map_location="cpu")
            assert isinstance(state_dict, dict), "state_dict must be a loaded dict or a valid checkpoint path"
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model


class Normalize(nn.Module):
    def __init__(self, dim):
        super(Normalize, self).__init__()
        self.norm2d = LayerNorm(dim)

    def forward(self, inputs):
        inputs = torch.as_tensor(inputs).float()
        inputs = inputs.view(-1, inputs.shape[-2], inputs.shape[-1])
        output = self.norm2d(inputs)
        return output


def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)


def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config


class TACNet(TACNetPreTrainedModel):
    def __init__(self,  audio_config1,audio_config2,audio_config3, task_config):
        super(TACNet, self).__init__(audio_config1,audio_config2,audio_config3)
        self.task_config = task_config
        self.num_classes = task_config.num_classes
        self.aligned = task_config.aligned
        self.proto_m = task_config.proto_m


        audio_config1 = update_attr("audio_config1", audio_config1, "num_hidden_layers",
                                   self.task_config, "audio1_num_hidden_layers")
        self.audio1 = TfModel(audio_config1)
        audio_config2 = update_attr("audio_config2", audio_config2, "num_hidden_layers",
                                    self.task_config, "audio2_num_hidden_layers")
        self.audio2 = TfModel(audio_config2)
        audio_config3 = update_attr("audio_config3", audio_config3, "num_hidden_layers",
                                    self.task_config, "audio3_num_hidden_layers")
        self.audio3 = TfModel(audio_config3)
        self.audio1_norm = Normalize(task_config.audio1_dim)
        self.audio2_norm = Normalize(task_config.audio2_dim)
        self.audio3_norm = Normalize(task_config.audio3_dim)




        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.criterion_cl = SupConLoss()

        self.apply(self.init_weights)


        self.audio_attention1 = MLAttention(self.num_classes, task_config.hidden_size)
        self.audio_attention2 = LocalGlobalAttention(self.num_classes, task_config.hidden_size)
        self.audio_attention3 = TimeFreqAttention(self.num_classes, task_config.hidden_size)
        self.proj_audio1 = MLLinear([task_config.hidden_size, task_config.hidden_size // 2], task_config.proj_size)
        self.proj_audio2 = MLLinear([task_config.hidden_size, task_config.hidden_size // 2], task_config.proj_size)
        self.proj_audio3 = MLLinear([task_config.hidden_size, task_config.hidden_size // 2], task_config.proj_size)

        self.de_proj_audio1 = MLLinear([task_config.proj_size, task_config.hidden_size // 2], task_config.hidden_size)
        self.de_proj_audio2 = MLLinear([task_config.proj_size, task_config.hidden_size // 2], task_config.hidden_size)
        self.de_proj_audio3 = MLLinear([task_config.proj_size, task_config.hidden_size // 2], task_config.hidden_size)

        self.a1a2toa3 = MLLinear([task_config.hidden_size * 3], task_config.hidden_size)
        self.a1a3toa2 = MLLinear([task_config.hidden_size * 3], task_config.hidden_size)
        self.a2a3toa1 = MLLinear([task_config.hidden_size * 3], task_config.hidden_size)
        self.max_pool = nn.MaxPool1d(3)

        self.agg = MLLinear([task_config.hidden_size * self.num_classes, task_config.hidden_size], self.num_classes)

        self.audio1_clf_weight = nn.Parameter(torch.Tensor(self.num_classes, task_config.hidden_size))
        nn.init.kaiming_uniform_(self.audio1_clf_weight, a=math.sqrt(5))
        self.audio2_clf_weight = nn.Parameter(torch.Tensor(self.num_classes, task_config.hidden_size))
        nn.init.kaiming_uniform_(self.audio2_clf_weight, a=math.sqrt(5))
        self.audio3_clf_weight = nn.Parameter(torch.Tensor(self.num_classes, task_config.hidden_size))
        nn.init.kaiming_uniform_(self.audio3_clf_weight, a=math.sqrt(5))

        self.sigmoid = nn.Sigmoid()

        self.register_buffer('audio1_pos_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('audio1_neg_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('audio2_pos_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('audio2_neg_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('audio3_pos_protos', torch.zeros(self.num_classes, task_config.proj_size))
        self.register_buffer('audio3_neg_protos', torch.zeros(self.num_classes, task_config.proj_size))

        self.register_buffer('queue', torch.randn(task_config.moco_queue, task_config.proj_size))
        self.register_buffer("queue_label", torch.randn(task_config.moco_queue, 1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=0)

        if not self.aligned:
            self.a2t_ctc = CTCModule(task_config.audio_dim, 50 if task_config.unaligned_mask_same_length else 500)
            self.v2t_ctc = CTCModule(task_config.audio_dim, 50 if task_config.unaligned_mask_same_length else 500)

    def dequeue_and_enqueue(self, feats, labels):
        batch_size = feats.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size >= self.task_config.moco_queue:
            self.queue[ptr:, :] = feats[:self.task_config.moco_queue - ptr, :]
            self.queue[:batch_size - self.task_config.moco_queue + ptr, :] = feats[self.task_config.moco_queue - ptr:,
                                                                             :]
            self.queue_label[ptr:, :] = labels[:self.task_config.moco_queue - ptr, :]
            self.queue_label[:batch_size - self.task_config.moco_queue + ptr, :] = labels[
                                                                                   self.task_config.moco_queue - ptr:,
                                                                                   :]
        else:
            self.queue[ptr:ptr + batch_size, :] = feats
            self.queue_label[ptr:ptr + batch_size, :] = labels
        ptr = (ptr + batch_size) % self.task_config.moco_queue  # move pointer
        self.queue_ptr[0] = ptr




    def get_all_audio_output(self, audio1, audio1_mask, audio2, audio2_mask, audio3, audio3_mask):
        audio1_layers, audio1_pooled_output = self.audio1(audio1, audio1_mask, output_all_encoded_layers=True)
        audio1_output = audio1_layers[-1]
        audio2_layers, audio2_pooled_output = self.audio2(audio2, audio2_mask, output_all_encoded_layers=True)
        audio2_output = audio2_layers[-1]
        audio3_layers, audio3_pooled_output = self.audio3(audio3, audio3_mask, output_all_encoded_layers=True)
        audio3_output = audio3_layers[-1]
        return audio1_output,audio2_output,audio3_output

    def get_cl_labels(self, labels, times=1):

        audio1_labels = torch.zeros_like(labels) + labels
        audio2_labels = torch.zeros_like(labels) + labels
        audio3_labels = torch.zeros_like(labels) + labels
        audio1_cl_labels = torch.zeros_like(audio1_labels, dtype=torch.long)
        audio2_cl_labels = torch.zeros_like(audio2_labels, dtype=torch.long)
        audio3_cl_labels = torch.zeros_like(audio3_labels, dtype=torch.long)

        example_idx, label_idx = torch.where(audio1_labels >= 0.5)
        audio1_cl_labels[example_idx, label_idx] = label_idx
        example_idx, label_idx = torch.where(audio1_labels < 0.5)
        audio1_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 1

        example_idx, label_idx = torch.where(audio2_labels >= 0.5)
        audio2_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 2
        example_idx, label_idx = torch.where(audio2_labels < 0.5)
        audio2_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 3

        example_idx, label_idx = torch.where(audio3_labels >= 0.5)
        audio3_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 4
        example_idx, label_idx = torch.where(audio3_labels < 0.5)
        audio3_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 5

        cl_labels = torch.stack([audio1_cl_labels, audio2_labels, audio3_cl_labels], dim=1)

        cl_labels = cl_labels.to(torch.int)
        if times > 1:
            final_cl_labels = torch.cat([cl_labels, cl_labels], dim=1)
            for i in range(2, times):
                final_cl_labels = torch.cat([final_cl_labels, cl_labels], dim=1)
        else:
            final_cl_labels = cl_labels
        return final_cl_labels

    def get_cl_mask(self, cl_labels, batch_size):
        mask = torch.eq(cl_labels[:batch_size], cl_labels.T).float()
        neg_mask = torch.ones_like(mask)
        return mask, neg_mask

    def update_protos(self, pos_protos, neg_protos, feats, gt_labels):
        b, c = gt_labels.shape[0], gt_labels.shape[1]
        for i in range(b):
            for j in range(c):
                if gt_labels[i][j] == 1:
                    pos_protos[j] = pos_protos[j] * self.proto_m + (1 - self.proto_m) * feats[i][j]
                else:
                    neg_protos[j] = neg_protos[j] * self.proto_m + (1 - self.proto_m) * feats[i][j]

    def forward(self, audio1, audio1_mask, audio2, audio2_mask, audio3, audio3_mask,
                label_input, label_mask, groundTruth_labels=None, training=True):

        audio1 = self.audio1_norm(audio1)
        audio2 = self.audio2_norm(audio2)
        audio3 = self.audio3_norm(audio3)

        '''
        if self.aligned == False:
            visual, v2t_position = self.v2t_ctc(visual)
            audio, a2t_position = self.a2t_ctc(audio)
            '''
        audio1_output, audio2_output, audio3_output = self.get_all_audio_output( audio1, audio1_mask, audio2, audio2_mask, audio3, audio3_mask)  # [B, L, D]
        audio1_lsr, audio_attention1 = self.audio_attention1(audio1_output, (1 - audio1_mask).type(torch.bool))
        audio2_lsr, audio_attention2 = self.audio_attention2(audio2_output, (1 - audio2_mask).type(torch.bool))
        audio3_lsr, audio_attention3 = self.audio_attention3(audio3_output, (1 - audio3_mask).type(torch.bool))

        latent_audio1 = self.proj_audio1(audio1_lsr)
        latent_audio2 = self.proj_audio2(audio2_lsr)
        latent_audio3 = self.proj_audio3(audio3_lsr)
        recon_audio1 = self.de_proj_audio1(latent_audio1)
        recon_audio2 = self.de_proj_audio2(latent_audio2)
        recon_audio3 = self.de_proj_audio3(latent_audio3)

        audio1_n = F.normalize(latent_audio1, p=2, dim=-1)
        audio2_n = F.normalize(latent_audio2, p=2, dim=-1)
        audio3_n = F.normalize(latent_audio3, p=2, dim=-1)
        audio1_protos = torch.stack([self.audio1_pos_protos, self.audio1_neg_protos])
        audio2_protos = torch.stack([self.audio2_pos_protos, self.audio2_neg_protos])
        audio3_protos = torch.stack([self.audio3_pos_protos, self.audio3_neg_protos])
        audio1_sim = torch.einsum('bld,nld->bln', audio1_n, audio1_protos)
        audio2_sim = torch.einsum('bld,nld->bln', audio2_n, audio2_protos)
        audio3_sim = torch.einsum('bld,nld->bln', audio3_n, audio3_protos)
        audio1_sim = torch.softmax(audio1_sim, dim=-1)
        audio2_sim = torch.softmax(audio2_sim, dim=-1)
        audio3_sim = torch.softmax(audio3_sim, dim=-1)

        if not training:
            audio1_pos_sim, audio1_neg_sim = audio1_sim[:, :, 0], audio1_sim[:, :, 1]
            audio2_pos_sim, audio2_neg_sim = audio2_sim[:, :, 0], audio2_sim[:, :, 1]
            audio3_pos_sim, audio3_neg_sim = audio3_sim[:, :, 0], audio3_sim[:, :, 1]
            audio1_pos_mask = (audio1_pos_sim > audio1_neg_sim).to(torch.float)
            audio1_neg_mask = 1 - audio1_pos_mask
            audio2_pos_mask = (audio2_pos_sim > audio2_neg_sim).to(torch.float)
            audio2_neg_mask = 1 - audio2_pos_mask
            audio3_pos_mask = (audio3_pos_sim > audio3_neg_sim).to(torch.float)
            audio3_neg_mask = 1 - audio3_pos_mask
            audio1_latent_padding = audio1_pos_mask.unsqueeze(-1) * self.audio1_pos_protos.unsqueeze(0) + \
                                    audio1_neg_mask.unsqueeze(-1) * self.audio1_neg_protos.unsqueeze(0)
            audio2_latent_padding = audio2_pos_mask.unsqueeze(-1) * self.audio2_pos_protos.unsqueeze(0) + \
                                    audio2_neg_mask.unsqueeze(-1) * self.audio2_neg_protos.unsqueeze(0)
            audio3_latent_padding = audio3_pos_mask.unsqueeze(-1) * self.audio3_pos_protos.unsqueeze(0) + \
                                   audio3_neg_mask.unsqueeze(-1) * self.audio3_neg_protos.unsqueeze(0)
        else:
            audio1_latent_padding = torch.einsum('bln,nld->bld', audio1_sim, audio1_protos)
            audio2_latent_padding = torch.einsum('bln,nld->bld', audio2_sim, audio2_protos)
            audio3_latent_padding = torch.einsum('bln,nld->bld', audio3_sim, audio3_protos)
        audio1_padding = self.de_proj_audio1(audio1_latent_padding)
        audio2_padding = self.de_proj_audio2(audio2_latent_padding)
        audio3_padding = self.de_proj_audio3(audio3_latent_padding)

        audio3_aug = self.a1a2toa3(torch.cat([recon_audio1, recon_audio2, audio3_padding], dim=-1))
        audio2_aug = self.a1a3toa2(torch.cat([recon_audio1, audio2_padding, recon_audio3], dim=-1))
        audio1_aug = self.a2a3toa1(torch.cat([audio1_padding, recon_audio2, recon_audio3], dim=-1))
        audio1_clf_out_3 = torch.einsum('bld,ld->bl', audio1_aug, self.audio1_clf_weight)
        audio2_clf_out_3 = torch.einsum('bld,ld->bl', audio2_aug, self.audio2_clf_weight)
        audio3_clf_out_3 = torch.einsum('bld,ld->bl', audio3_aug, self.audio3_clf_weight)

        audio3_beta = self.a1a2toa3(torch.cat([audio1_aug, audio2_aug, audio3_aug], dim=-1))
        audio2_beta = self.a1a3toa2(torch.cat([audio1_aug, audio2_aug, audio3_aug], dim=-1))
        audio1_beta = self.a2a3toa1(torch.cat([audio1_aug, audio2_aug, audio3_aug], dim=-1))
        audio1_clf_out_4 = torch.einsum('bld,ld->bl', audio1_beta, self.audio1_clf_weight)
        audio2_clf_out_4 = torch.einsum('bld,ld->bl', audio2_beta, self.audio2_clf_weight)
        audio3_clf_out_4 = torch.einsum('bld,ld->bl', audio3_beta, self.audio3_clf_weight)

        audio1_clf_out_1 = torch.einsum('bld,ld->bl', audio1_lsr, self.audio1_clf_weight)
        audio2_clf_out_1 = torch.einsum('bld,ld->bl', audio2_lsr, self.audio2_clf_weight)
        audio3_clf_out_1 = torch.einsum('bld,ld->bl', audio3_lsr, self.audio3_clf_weight)

        if training:
            latent_aug_audio1 = self.proj_audio1(audio1_aug)
            latent_aug_audio2 = self.proj_audio2(audio2_aug)
            latent_aug_audio3 = self.proj_audio3(audio3_aug)
            latent_beta_audio1 = self.proj_audio1(audio1_beta)
            latent_beta_audio2 = self.proj_audio2(audio2_beta)
            latent_beta_audio3 = self.proj_audio3(audio3_beta)
            total_proj = torch.stack([latent_audio1, latent_audio2, latent_audio3,
                                      latent_aug_audio1, latent_aug_audio2, latent_aug_audio3,
                                      latent_beta_audio1, latent_beta_audio2, latent_beta_audio3], dim=1)
            label_time = 3

            total_proj = total_proj.view(-1, total_proj.shape[-1])
            total_proj = F.normalize(total_proj, dim=-1)
            cl_labels = self.get_cl_labels(groundTruth_labels, times=label_time).view(-1).unsqueeze(-1)
            audio1_norm = F.normalize(latent_audio1.data, dim=-1)
            audio2_norm = F.normalize(latent_audio2.data, dim=-1)
            audio3_norm = F.normalize(latent_audio3.data, dim=-1)
            cl_feats = torch.cat((total_proj, self.queue.clone().detach()), dim=0)
            total_cl_labels = torch.cat((cl_labels, self.queue_label.clone().detach()), dim=0)
            batch_size = cl_feats.shape[0]
            cl_mask, cl_neg_mask = self.get_cl_mask(total_cl_labels, batch_size)
            cl_loss = self.criterion_cl(cl_feats, cl_mask, cl_neg_mask, batch_size)
            self.dequeue_and_enqueue(total_proj, cl_labels)
            self.update_protos(self.audio1_pos_protos, self.audio1_neg_protos, audio1_norm, groundTruth_labels)
            self.update_protos(self.audio2_pos_protos, self.audio2_neg_protos, audio2_norm, groundTruth_labels)
            self.update_protos(self.audio3_pos_protos, self.audio3_neg_protos, audio3_norm, groundTruth_labels)
        # predict_scores_text4 = self.sigmoid(text_clf_out_4)
        # predict_scores_visual4 = self.sigmoid(visual_clf_out_4)
        # predict_scores_audio4 = self.sigmoid(audio_clf_out_4)
        # # predict_scores_mean = (predict_scores_text4 + predict_scores_visual4 + predict_scores_audio4) / 3

        clf_out_1 = torch.stack([audio1_clf_out_1, audio2_clf_out_1, audio3_clf_out_1], dim=-1)
        clf_out_1 = self.max_pool(clf_out_1).squeeze(-1)
        clf_out_3 = torch.stack([audio1_clf_out_3, audio2_clf_out_3, audio3_clf_out_3], dim=-1)
        clf_out_3 = self.max_pool(clf_out_3).squeeze(-1)
        clf_out_4 = torch.stack([audio1_clf_out_4, audio2_clf_out_4, audio3_clf_out_4], dim=-1)
        clf_out_4 = self.max_pool(clf_out_4).squeeze(-1)
        predict_scores_clf4 = self.sigmoid(clf_out_4)
        # predict_labels_clf4 = getBinaryTensor(predict_scores_clf4, boundary=self.task_config.binary_threshold)
        # max_scores = torch.stack([predict_scores_clf4, predict_scores_clf4, ])
        # max_labels = torch.stack([predict_labels_clf4, predict_labels_clf4, ])
        # predict_labels = torch.stack([max_labels, max_labels], dim=0)
        # predict_scores = torch.stack([max_scores, max_scores], dim=0)

        total_aug = torch.stack([audio1_beta, audio2_beta, audio3_beta], dim=1)
        agg_out = self.agg(total_aug.view(total_aug.shape[0], total_aug.shape[1], -1))
        agg_scores = self.sigmoid(agg_out)
        predict_agg_scores = torch.mean(agg_scores, dim=1)
        # predict_agg_labels = getBinaryTensor(predict_agg_scores, boundary=self.task_config.binary_threshold)
        # predict_labels = torch.cat([predict_labels, predict_agg_labels.unsqueeze(0)], dim=0)
        # predict_scores = torch.cat([predict_scores, predict_agg_scores.unsqueeze(0)], dim=0)
        # predict_agg_scores_mean = (predict_agg_scores + predict_scores_mean) / 2
        # predict_agg_labels_mean = getBinaryTensor(predict_agg_scores_mean, boundary=self.task_config.binary_threshold)
        # predict_labels = torch.cat([predict_labels, predict_agg_labels_mean.unsqueeze(0)], dim=0)
        # predict_scores = torch.cat([predict_scores, predict_agg_scores_mean.unsqueeze(0)], dim=0)

        predict_final_scores_mean = (predict_agg_scores + predict_scores_clf4) / 2
        predict_final_labels_mean = getBinaryTensor(predict_final_scores_mean,
                                                    boundary=self.task_config.binary_threshold)
        predict_scores = predict_final_scores_mean
        predict_labels = predict_final_labels_mean
        # predict_scores = torch.cat([predict_scores, predict_final_scores_mean.unsqueeze(0)], dim=0)

        if training:
            total_aug_clf_loss = self.bce_loss(agg_out, groundTruth_labels.unsqueeze(-2).repeat(1, 3, 1))
            shuffle_sample_idx = torch.zeros(self.num_classes, total_aug.shape[1], total_aug.shape[0], dtype=torch.long)
            for l in range(self.num_classes):
                for m in range(total_aug.shape[1]):
                    one_idx = np.random.permutation(total_aug.shape[0])
                    shuffle_sample_idx[l][m] += one_idx
            shuffle_sample_idx = shuffle_sample_idx.permute(2, 1, 0)

            shuffle_modality_idx = torch.zeros(self.num_classes, total_aug.shape[0], total_aug.shape[1],
                                               dtype=torch.long)
            for l in range(self.num_classes):
                for s in range(total_aug.shape[0]):
                    one_idx = np.random.permutation(total_aug.shape[1])
                    shuffle_modality_idx[l][s] += one_idx
            shuffle_modality_idx = shuffle_modality_idx.permute(1, 2, 0)

            label_idx = torch.zeros(total_aug.shape[0], total_aug.shape[1], self.num_classes) + torch.tensor(
                list(range(self.num_classes)))
            label_idx = label_idx.to(torch.long)
            shuffle_total_aug = total_aug[shuffle_sample_idx, shuffle_modality_idx, label_idx]
            shuffle_aug_out = self.agg(shuffle_total_aug.view(total_aug.shape[0], total_aug.shape[1], -1))
            shuffle_gt_labels = groundTruth_labels.unsqueeze(-2).repeat(1, 3, 1)[
                shuffle_sample_idx, shuffle_modality_idx, label_idx]
            shuffle_aug_clf_loss = self.bce_loss(shuffle_aug_out, shuffle_gt_labels)

        if training:
            all_loss = 0
            clf_loss = self.bce_loss(clf_out_1, groundTruth_labels) * self.task_config.lsr_clf_weight
            clf_loss += self.bce_loss(clf_out_3, groundTruth_labels) * self.task_config.aug_clf_weight
            clf_loss += self.bce_loss(clf_out_4, groundTruth_labels)
            all_loss += clf_loss

            all_loss += cl_loss * self.task_config.cl_weight

            aug_mse_loss = self.mse_loss(audio1_aug, audio1_lsr) + self.mse_loss(audio2_aug, audio2_lsr) \
                           + self.mse_loss(audio3_aug, audio3_lsr)
            beta_mse_loss = self.mse_loss(audio1_beta, audio1_lsr) + self.mse_loss(audio2_beta, audio2_lsr) \
                            + self.mse_loss(audio3_beta, audio3_lsr)
            recon_mse_loss = self.mse_loss(recon_audio1, audio1_lsr) + self.mse_loss(recon_audio2, audio2_lsr) \
                             + self.mse_loss(recon_audio3, audio3_lsr)
            all_loss += recon_mse_loss * self.task_config.recon_mse_weight \
                        + aug_mse_loss * self.task_config.aug_mse_weight + beta_mse_loss * self.task_config.beta_mse_weight

            all_loss += total_aug_clf_loss * self.task_config.total_aug_clf_weight
            all_loss += shuffle_aug_clf_loss * self.task_config.shuffle_aug_clf_weight
            return all_loss, predict_labels, groundTruth_labels, predict_scores
        else:

            return predict_labels, groundTruth_labels, predict_scores


