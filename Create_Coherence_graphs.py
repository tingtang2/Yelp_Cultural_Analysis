#!/usr/bin/env python
# coding: utf-8

# In[3]:


from matplotlib import pyplot as plt


# # Create plots for paper

# In[41]:


fig, ax = plt.subplots(2,3, figsize=(18, 10))
x_vals = [5, 10, 25, 50]

c3rLDA_npmi = [0.185, 0.196, 0.265, 0.057]
scholar_npmi = [0.271, 0.249, 0.21, 0.18]
ccLDA_npmi = [0.163, 0.179, 0.132, 0.12]

ax[0, 0].plot(x_vals, c3rLDA_npmi, marker= 'X', ls='-', c='g')
ax[0, 0].plot(x_vals, scholar_npmi, marker='X', ls='-', c='b')
ax[0, 0].plot(x_vals, ccLDA_npmi, marker='X', ls='-', c='r')
ax[0, 0].set_xticks(x_vals)
ax[0, 0].set_xlabel('# of Topics')
ax[0, 0].set_ylabel('NPMI-internal')

c3rLDA_npmi = [0.032, 0.014, -0.062, -0.012]
scholar_npmi = [-0.093,-0.105,-0.093,-0.0867]
ccLDA_npmi = [-0.028, 0.0219, -0.054, -0.064]


l1, = ax[0, 1].plot(x_vals, c3rLDA_npmi, marker= 'X', ls='-', c='g')
l2, = ax[0, 1].plot(x_vals, scholar_npmi,marker= 'X', ls='-', c='b')
l3, = ax[0, 1].plot(x_vals, ccLDA_npmi, marker='X', ls='-', c='r')
ax[0, 1].set_xticks(x_vals)
ax[0, 1].set_xlabel('# of Topics')
ax[0, 1].set_ylabel('NPMI-external')


c3rLDA_ca = [0.174, 0.168, 0.142, 0.016]
scholar_ca = [0.087, 0.094, 0.102, 0.097]
ccLDA_ca = [0.145, 0.146, 0.113, 0.111]


ax[0, 2].plot(x_vals, c3rLDA_ca, marker= 'X', ls='-', c='g')
ax[0, 2].plot(x_vals, scholar_ca, marker= 'X', ls='-', c='b')
ax[0, 2].plot(x_vals, ccLDA_ca, marker= 'X', ls='-', c='r')
ax[0, 2].set_xlabel('# of Topics')
ax[0, 2].set_xticks(x_vals)
ax[0, 2].set_ylabel('C_A')

c3rLDA_cp = [0.446, 0.263, -0.211, -0.22]
scholar_cp = [-0.506, -0.492, -0.476, -0.444]
ccLDA_cp = [-0.152, -0.083, -0.266, -0.287]


ax[1, 0].plot(x_vals, c3rLDA_cp, marker= 'X', ls='-', c='g')
ax[1, 0].plot(x_vals, scholar_cp, marker= 'X', ls='-', c='b')
ax[1, 0].plot(x_vals, ccLDA_cp, marker= 'X', ls='-', c='r')
ax[1, 0].set_xlabel('# of Topics')
ax[1, 0].set_xticks(x_vals)
ax[1, 0].set_ylabel('C_P')

c3rLDA_cv = [0.523, 0.515, 0.526, 0.2]
scholar_cv = [0.522, 0.494, 0.479, 0.465]
ccLDA_cv = [0.47, 0.46, 0.454, 0.45]


ax[1, 1].plot(x_vals, c3rLDA_cv, marker= 'X', ls='-', c='g')
ax[1, 1].plot(x_vals, scholar_cv, marker= 'X', ls='-', c='b')
ax[1, 1].plot(x_vals, ccLDA_cv, marker= 'X', ls='-', c='r')
ax[1, 1].set_xlabel('# of Topics')
ax[1, 1].set_xticks(x_vals)
ax[1, 1].set_ylabel('C_V')

c3rLDA_umass = [-4.347, -4.682, -7, -2.112]
scholar_umass = [-8.817, -8.227, -7.608, -6.951]
ccLDA_umass = [-5.58, -5.243, -5.952, -5.938]


ax[1, 2].plot(x_vals, c3rLDA_umass, marker= 'X', ls='-', c='g')
ax[1, 2].plot(x_vals, scholar_umass, marker= 'X', ls='-', c='b')
ax[1, 2].plot(x_vals, ccLDA_umass, marker= 'X', ls='-', c='r')
ax[1, 2].set_xlabel('# of Topics')
ax[1, 2].set_xticks(x_vals)
ax[1, 2].set_ylabel('UMass')


fig.suptitle(r'Cuisine Topic Coherence vs. Number of Topics')

plt.figlegend((l1, l2, l3), ('c3rLDA', 'Scholar', 'ccLDA'), loc='lower center', shadow=True)

#plt.tight_layout()
#plt.show()
plt.savefig('cuisine_topic_coherence_vs_topics.png', bbox_inches = "tight")

