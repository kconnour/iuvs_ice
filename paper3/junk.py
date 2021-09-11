import glob
import numpy as np
import matplotlib.pyplot as plt
from paper3.txt_data import L2Txt
from mpl_toolkits.axes_grid1 import make_axes_locatable

dust = np.load('/home/kyle/cloud_retrievals/dust_test.npy')
ice = np.load('/home/kyle/cloud_retrievals/ice_test.npy')

l2_files = sorted(glob.glob('/media/kyle/Samsung_T5/l2txt/orbit03400/*3453*'))
l2_file = L2Txt(l2_files[5])
print(l2_file.tau_dust.shape)
#print(dust[:10, :, 1])
#print(ice[:10, :, 1])

fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
img = ax.imshow(dust[:120, :, 1], cmap='viridis', vmin=0, vmax=0.5)
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(img, cax=cax, orientation='vertical')
plt.savefig('/home/kyle/cloud_retrievals/dust_test.png', dpi=300)

fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
img = ax.imshow(ice[:120, :, 1], cmap='viridis', vmin=0, vmax=5)
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(img, cax=cax, orientation='vertical')
plt.savefig('/home/kyle/cloud_retrievals/ice_test.png', dpi=300)

# Franck's dust
fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
img = ax.imshow(l2_file.tau_dust[:120, :], cmap='viridis', vmin=0, vmax=0.5)
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(img, cax=cax, orientation='vertical')
plt.savefig('/home/kyle/cloud_retrievals/dust_franck_test.png', dpi=300)

# Franck's albedo
fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
img = ax.imshow(l2_file.albedo[:120, :], cmap='viridis', vmin=0, vmax=1)
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(img, cax=cax, orientation='vertical')
plt.savefig('/home/kyle/cloud_retrievals/franck_albedo.png', dpi=300)

# Ozone
fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
img = ax.imshow(l2_file.o3[:120, :], cmap='viridis', vmin=0, vmax=5)
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(img, cax=cax, orientation='vertical')
plt.savefig('/home/kyle/cloud_retrievals/franck_o3.png', dpi=300)


