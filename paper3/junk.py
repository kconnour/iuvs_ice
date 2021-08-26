import numpy as np
import matplotlib.pyplot as plt

dust = np.load('/home/kyle/cloud_retrievals/dust_test.npy')
ice = np.load('/home/kyle/cloud_retrievals/ice_test.npy')

print(dust[:10, :, 1])
print(ice[:10, :, 1])
plt.imshow(dust[:, :, 1], cmap='viridis', vmin=0, vmax=3)
plt.savefig('/home/kyle/cloud_retrievals/dust0.png', dpi=300)

plt.imshow(ice[:, :, 1], cmap='viridis', vmin=0, vmax=3)
plt.savefig('/home/kyle/cloud_retrievals/ice0.png', dpi=300)
