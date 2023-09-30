import torch.nn as nn
import torch
import torch.nn.functional as F

class Encoder_MLP(nn.Module):
    def __init__(self, D=5, W=256, input_ch=3, input_ch_color=3, input_ch_message=4, input_ch_views=3, output_ch=4,
                 skips=[-1], use_viewdirs=False):
        """
        """
        super(Encoder_MLP, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_color = input_ch_color
        self.input_ch_message = input_ch_message
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch + input_ch_color + input_ch_message, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + input_ch_color + input_ch_message + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, rgbsigma, message):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        message = message.expand(input_pts.shape[0], -1)
        h = torch.cat([input_pts, rgbsigma, message], -1)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views, rgbsigma, message], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, rgbsigma], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


class Encoder(torch.nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, D, H, W, in_channels=3, encoder_channels=64, encoder_blocks=4, message_dim=4):
        super(Encoder, self).__init__()
        self.D = D
        self.H = H
        self.W = W
        self.conv_channels = encoder_channels
        self.in_channels = in_channels
        self.num_blocks = encoder_blocks

        layers = [Conv3dBNRelu(self.in_channels, self.conv_channels)]

        for _ in range(encoder_blocks-1):
            layer = Conv3dBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv3d_layers = torch.nn.Sequential(*layers)
        self.after_concat_layer = Conv3dBNRelu(self.conv_channels + self.in_channels + message_dim,
                                             self.conv_channels)

        self.final_layer = torch.nn.Conv3d(self.conv_channels, 3, kernel_size=1)

    def forward(self, image, weights, message):

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1).unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1, -1, self.D, self.H, self.W)

        N_rays, N_samples, channels = image.shape
        x = image.unsqueeze(0).permute(0, 3, 2, 1).reshape(1, channels, N_samples, self.H, self.W).contiguous()
        y = weights.unsqueeze(0).unsqueeze(-1).permute(0, 3, 2, 1).reshape(1, 1, self.D, self.H, self.W).contiguous()
        input = torch.cat([x, y], dim=1)

        encoded_image = self.conv3d_layers(input)
        # concatenate expanded message and image
        concat = torch.cat([expanded_message, encoded_image, x, y], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        im_w = im_w.reshape(-1, channels, N_samples, N_rays).permute(0, 3, 2, 1).squeeze(0).contiguous()
        # im_w = im_w * 2.0 - 1.0
        return im_w


class Conv3dBNRelu(torch.nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, stride=1):
        super(Conv3dBNRelu, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv3d(channels_in, channels_out, 3, stride, padding=1),
            torch.nn.BatchNorm3d(channels_out),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Encoder_Tri_MLP(nn.Module):
    def __init__(self, D=3, W=256, input_ch=3, input_ch_color=3, input_ch_message=4, input_ch_views=3, output_ch=4,
                 skips=[-1], use_viewdirs=False, D_m=2, W_m=128):
        """
        """
        super(Encoder_Tri_MLP, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_color = input_ch_color
        self.input_ch_message = input_ch_message
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch + input_ch_color, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                range(D - 1)])

        self.feature_linears = nn.ModuleList(
            [nn.Linear(input_ch_message, W_m)] + [
                nn.Linear(W_m, W_m) if i not in self.skips else nn.Linear(W_m + input_ch, W_m) for i in
                range(D_m - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + input_ch_color +
                                                      W_m + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, rgbsigma, message):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        message = message.expand(input_pts.shape[0], -1)
        h = torch.cat([input_pts, rgbsigma], -1)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        message_feature = message
        for i, l in enumerate(self.feature_linears):
            message_feature = self.feature_linears[i](message_feature)
            message_feature = F.relu(message_feature)
            if i in self.skips:
                message_feature = torch.cat([input_pts, message_feature], -1)
        if self.use_viewdirs:
            # alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views, rgbsigma, message_feature], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, rgbsigma], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


class Encoder_Tri_MLP_add(nn.Module):
    def __init__(self, D=3, W=256, input_ch=3, input_ch_color=3, input_ch_message=4, input_ch_views=3, output_ch=4,
                 skips=[-1], use_viewdirs=False, D_m=2, W_m=128):
        """
        """
        super(Encoder_Tri_MLP_add, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_color = input_ch_color
        self.input_ch_message = input_ch_message
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.feature_color = nn.ModuleList(
            [nn.Linear(input_ch + input_ch_color, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                range(D - 1)])

        self.feature_message = nn.ModuleList(
            [nn.Linear(input_ch_message, W_m)] + [
                nn.Linear(W_m, W_m) if i not in self.skips else nn.Linear(W_m + input_ch, W_m) for i in
                range(D_m - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.feature_fusion = nn.ModuleList([nn.Linear(input_ch_views + input_ch_color +
                                                      W_m + W + input_ch_message, W), nn.Linear(W + input_ch_message, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, rgbsigma, message):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        message = message.expand(input_pts.shape[0], -1)
        h = torch.cat([input_pts, rgbsigma], -1)
        for i, l in enumerate(self.feature_color):
            h = self.feature_color[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        message_feature = message
        for i, l in enumerate(self.feature_message):
            message_feature = self.feature_message[i](message_feature)
            message_feature = F.relu(message_feature)
            if i in self.skips:
                message_feature = torch.cat([input_pts, message_feature], -1)
        if self.use_viewdirs:
            # alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views, rgbsigma, message_feature], -1)

            for i, l in enumerate(self.feature_fusion):
                h = torch.cat([h, message], -1)
                h = self.feature_fusion[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h) + rgbsigma[:, 0:3]
            outputs = torch.cat([rgb, rgbsigma], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

class Encoder_sigma(nn.Module):
    def __init__(self, D=3, W=256, input_ch=3, input_ch_color=3, input_ch_message=4, input_ch_views=3, output_ch=4,
                 skips=[-1], use_viewdirs=False, D_m=2, W_m=128):
        """
        """
        super(Encoder_sigma, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_color = input_ch_color
        self.input_ch_message = input_ch_message
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(1 + input_ch_message, W)] + [nn.Linear(W, W//2)] + [nn.Linear(W//2, 1)])

        # self.feature_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_message, W_m)] + [
        #         nn.Linear(W_m, W_m) if i not in self.skips else nn.Linear(W_m + input_ch, W_m) for i in
        #         range(D_m - 1)])
        #
        # ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + input_ch_color +
        #                                               W_m + W, W // 2)])
        #
        # ### Implementation according to the paper
        # # self.views_linears = nn.ModuleList(
        # #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        #
        # if use_viewdirs:
        #     self.feature_linear = nn.Linear(W, W)
        #     self.alpha_linear = nn.Linear(W, 1)
        #     self.rgb_linear = nn.Linear(W // 2, 3)
        # else:
        #     self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, rgbsigma, message):
        # input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        rgb, sigma = torch.split(rgbsigma, [3, 1], dim=-1)
        message = message.expand(sigma.shape[0], -1)
        h = torch.cat([sigma, message], -1)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)

        outputs = torch.cat([rgbsigma, h], -1)


        return outputs

class Encoder_Tri_MLP_f(nn.Module):
    def __init__(self, D=3, W=256, input_ch=3, input_ch_color=3, input_ch_message=4, input_ch_views=3, output_ch=4,
                 skips=[-1], use_viewdirs=False, D_m=2, W_m=128):
        """
        """
        super(Encoder_Tri_MLP_f, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_color = input_ch_color
        self.input_ch_message = input_ch_message
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.feature_color = nn.ModuleList(
            [nn.Linear(input_ch + input_ch_color, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                range(D - 1)])

        self.feature_message = nn.ModuleList(
            [nn.Linear(input_ch_message, W_m)] + [
                nn.Linear(W_m, W_m) if i not in self.skips else nn.Linear(W_m + input_ch, W_m) for i in
                range(D_m - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.feature_fusion = nn.ModuleList([nn.Linear(input_ch_views + input_ch_color +
                                                      W_m + W + input_ch_message + input_ch_color, W), 
                                                      nn.Linear(W + input_ch_message + input_ch_color, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, rgbsigma, message):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        message = message.expand(input_pts.shape[0], -1)
        h = torch.cat([input_pts, rgbsigma], -1)
        for i, l in enumerate(self.feature_color):
            h = self.feature_color[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        message_feature = message
        for i, l in enumerate(self.feature_message):
            message_feature = self.feature_message[i](message_feature)
            message_feature = F.relu(message_feature)
            if i in self.skips:
                message_feature = torch.cat([input_pts, message_feature], -1)
        if self.use_viewdirs:
            # alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views, rgbsigma, message_feature], -1)

            for i, l in enumerate(self.feature_fusion):
                h = torch.cat([h, message, rgbsigma], -1)
                h = self.feature_fusion[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, rgbsigma], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

class Encoder_wo_color(nn.Module):
    def __init__(self, D=3, W=256, input_ch=3, input_ch_color=3, input_ch_message=4, input_ch_views=3, output_ch=4,
                 skips=[-1], use_viewdirs=False, D_m=2, W_m=128):
        """
        """
        super(Encoder_wo_color, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_color = input_ch_color
        self.input_ch_message = input_ch_message
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # self.pts_linears = nn.ModuleList(
        #     [nn.Linear(input_ch + input_ch_color, W)] + [
        #         nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
        #         range(D - 1)])

        self.feature_linears = nn.ModuleList(
            [nn.Linear(input_ch_message, W_m)] + [
                nn.Linear(W_m, W_m) if i not in self.skips else nn.Linear(W_m + input_ch, W_m) for i in
                range(D_m - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_color + W_m , W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, rgbsigma, message):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        message = message.expand(input_pts.shape[0], -1)
        # h = torch.cat([input_pts, rgbsigma], -1)
        # for i, l in enumerate(self.pts_linears):
        #     h = self.pts_linears[i](h)
        #     h = F.relu(h)
        #     if i in self.skips:
        #         h = torch.cat([input_pts, h], -1)

        message_feature = message
        for i, l in enumerate(self.feature_linears):
            message_feature = self.feature_linears[i](message_feature)
            message_feature = F.relu(message_feature)
            if i in self.skips:
                message_feature = torch.cat([input_pts, message_feature], -1)

            h = torch.cat([rgbsigma, message_feature], -1)
        if self.use_viewdirs:
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, rgbsigma], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


class Encoder_wo_m(nn.Module):
    def __init__(self, D=3, W=256, input_ch=3, input_ch_color=3, input_ch_message=4, input_ch_views=3, output_ch=4,
                 skips=[-1], use_viewdirs=False, D_m=2, W_m=128):
        """
        """
        super(Encoder_wo_m, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_color = input_ch_color
        self.input_ch_message = input_ch_message
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch + input_ch_color, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                range(D - 1)])

        # self.feature_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_message, W_m)] + [
        #         nn.Linear(W_m, W_m) if i not in self.skips else nn.Linear(W_m + input_ch, W_m) for i in
        #         range(D_m - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views +
                                                      input_ch_message + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, rgbsigma, message):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        message = message.expand(input_pts.shape[0], -1)
        h = torch.cat([input_pts, rgbsigma], -1)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # message_feature = message
        # for i, l in enumerate(self.feature_linears):
        #     message_feature = self.feature_linears[i](message_feature)
        #     message_feature = F.relu(message_feature)
        #     if i in self.skips:
        #         message_feature = torch.cat([input_pts, message_feature], -1)
        if self.use_viewdirs:
            # alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views, message], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, rgbsigma], -1)
        else:
            outputs = self.output_linear(h)

        return outputs
