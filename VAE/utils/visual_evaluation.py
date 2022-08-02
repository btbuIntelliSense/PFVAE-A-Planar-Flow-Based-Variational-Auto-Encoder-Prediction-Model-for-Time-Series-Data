from __future__ import print_function
import os
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_reconstructions(data, recon_mean, loss, loss_type, epoch, args):

    # if epoch == 1:
    #     if not os.path.exists(args.snap_dir + 'reconstruction/'):
    #         os.makedirs(args.snap_dir + 'reconstruction/')
    plt.figure(figsize=(12, 4))
    plt.plot(data.reshape(-1, 1)[2000:3000])
    plt.plot(recon_mean.reshape(-1, 1)[2000:3000])
    plt.legend(['real', 'predict'], loc='best')
    plt.show()

    # if args.input_type == 'multinomial':
        # data is already between 0 and 1
        # num_classes = 256
        # Find largest class logit
        # tmp = recon_mean.view(-1, num_classes, *args.input_size).max(dim=1)[1]
        # recon_mean = tmp.float() / (num_classes - 1.)
#     if epoch == 1:
#         if not os.path.exists(args.snap_dir + 'reconstruction/'):
#             os.makedirs(args.snap_dir + 'reconstruction/')
#         # VISUALIZATION: plot real images
#         plot_images(args, data.data.cpu().numpy()[0:9], args.snap_dir + 'reconstruction/', 'real', # 如果是手写数字，这里是画真实的图像
#                     size_x=3, size_y=3)
#     # VISUALIZATION: plot reconstructions
#     if loss_type == 'bpd':
#         fname = str(epoch) + '_bpd_%5.3f' % loss
#     elif loss_type == 'elbo':
#         fname = str(epoch) + '_elbo_%6.4f' % loss
#     plot_images(args, recon_mean.data.cpu().numpy()[0:9], args.snap_dir + 'reconstruction/', fname, # 这里是画重构的图像
#                 size_x=3, size_y=3)
#
#
# def plot_images(args, x_sample, dir, file_name, size_x=3, size_y=3):
#
#     fig = plt.figure(figsize=(size_x, size_y))
#     # fig = plt.figure(1)
#     gs = gridspec.GridSpec(size_x, size_y)
#     gs.update(wspace=0.05, hspace=0.05)
#
#     for i, sample in enumerate(x_sample):
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         sample = sample.reshape((args.input_size[0], args.input_size[1], args.input_size[2]))
#         sample = sample.swapaxes(0, 2)
#         sample = sample.swapaxes(0, 1)
#         if (args.input_type == 'binary') or (args.input_type in ['multinomial'] and args.input_size[0] == 1):
#             sample = sample[:, :, 0]
#             plt.imshow(sample, cmap='gray', vmin=0, vmax=1)
#         else:
#             plt.imshow(sample)
#
#     plt.savefig(dir + file_name + '.png', bbox_inches='tight')
#     plt.close(fig)

