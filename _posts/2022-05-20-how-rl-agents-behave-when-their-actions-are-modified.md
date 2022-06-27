---
layout: post
title: "How RL Agents Behave When Their Actions Are Modified? [Distillation post]"
date: 2022-05-20 18:47:52 UTC
#subtitle: 
#background:
---

Summary
=======

This is a distillation post intended to summarize the article [How RL Agents Behave When Their Actions Are Modified?](https://www.aaai.org/AAAI21Papers/AAAI-5767.LangloisE.pdf) by Eric Langlois and Tom Everitt, published at AAAI-21. The article describes Modified Action MDPs, where the environment or another agent such as a human may override the action of an agent. Then it studies the behavior of different agents depending on the training objective. Interestingly, some standard agents may ignore the possibility of action modifications, making them corrigible.

Check also [this brief summary](https://www.alignmentforum.org/posts/Cd7Hw492RqooYgQAS/progress-on-causal-influence-diagrams#User_Interventions_and_Interruption) by the authors themselves. This post was corrected and improved thanks to comments by Eric Langlois.

Introduction
============

Causal incentives is one research line of AI Safety, sometimes framed as closely related to embedded agency, that aims to use causality to understand and model agent instrumental incentives. In what I consider a seminal paper, Tom Everitt et al. [[4](https://arxiv.org/abs/1902.09980)] showed how one may model instrumental incentives in a simple framework that unifies previous work on [AI oracles](https://www.lesswrong.com/tag/oracle-ai), interruptibility, and [corrigibility](https://www.lesswrong.com/tag/corrigibility). Indeed, while this research area makes strong assumptions about the agent or the environment it is placed, it goes straight to the heart of outer alignment and relates to embedded agents as we will see.

Since the paper, this research line has been quite productive, exploring multi-agent and multi-decision settings, its application to causal fairness, as well as more formally establishing causal definitions and diagrams that model the agent incentives (for more details check [causalincentives.com](https://causalincentives.com/)). In this paper, the authors build upon the definition of Response Incentive by Ryan Carey et al. [[2](https://arxiv.org/abs/2102.01685)] to study how popular Reinforcement Learning algorithms respond to a human that corrects the agent behavior.

Technical section
=================

Definitions
-----------

### Markov Decision Process

To explain the article results, the first step is to provide the definitions we will be using. In Reinforcement Learning, the environment  is almost always considered a **Markov Decision Process** (MDP) defined as the tuple \\(\\mathcal{M} = (\\mathcal{S}, \\mathcal{A},  \\mathcal{P}_S,\\mathcal{R}, \\gamma)\\), where \\(\\mathcal{S}\\) is the space of states, \\(\\mathcal{A}\\) the space of actions, \\(\\mathcal{P}_S: \\mathcal{S} \\times \\mathcal{A} \\rightarrow \\mathcal{S}\\) a function determining the transition probabilities, \\(\\mathcal{R}: \\mathcal{S} \\times \\mathcal{A} \\rightarrow \\mathbb{R}\\) the reward function, and \\(\\gamma\\) a temporal discount.

### Modified Action Markov Decision Process

In this article, however, the MDP definition is extended by adding an additional term \\(\\mathcal{P}_A\\) that represents how the agent's actions might be overridden by special circumstances not contemplated in the environment state transitions, for example by a human or due to physical constraints. \\(\\mathcal{P}_A\\) will depend not only on the state of the environment but also on the agent policy \\(\\Pi\\), \\(\\mathcal{P}_A: \\Pi \\times \\mathcal{S} \\rightarrow \\mathcal{A}\\). The environment is now considered a **Modified Action Markov Decision Process** (MAMDP), and we will denote it by \\(\\tilde{\\mathcal{M}} = (\\mathcal{M}, \\mathcal{P}_A)\\).

There is a very important difference between MDPs and MAMDPs: in MAMDPs, \\(\\mathcal{P}_A\\) is considered separately from the environment state transitions \\(\\mathcal{P}_S\\), and importantly may depend on the agent policy \\(\\pi\\). This means that the agent is now closer to becoming [an embedded agent](https://www.alignmentforum.org/s/Rm6oQRJJmhGCcLvxh), although it is not fully yet because modifications affect only particular actions, not the agent policy itself.

### Causal Influence Diagrams

To analyze the behavior of different RL algorithms in MAMDPs, Langlois and Everitt also use **Causal Influence Diagrams (**CIDs). These diagrams are Directed Acyclic Graphs where nodes represent probability distributions of a given variable and edges represent functions, and where we may define causal interventions by fixing the value of a node and eliminating incoming arrows (eg a [Structural Causal Model](https://en.wikipedia.org/wiki/Causal_model)). Furthermore, some nodes might be Decision (purple squares) or Utility (yellow rotated squares) nodes.

<figure class="image"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/61e0f13fb1ef16a67cc27d4ac8c9b491748fc3143a45c001.png"/><figcaption>Left: simple Causal Influence Diagram, the (optionally) punctuated edge represents information flow. Right: intervention on the CID to set the value of <span><span class="mjpage"><span class="mjx-chtml"><span aria-label="D" class="mjx-math"><span aria-hidden="true" class="mjx-mrow"><span class="mjx-mi"><span class="mjx-char MJXc-TeX-math-I" style="padding-top: 0.446em; padding-bottom: 0.298em;">D</span></span></span></span></span></span></span> to <span><span class="mjpage"><span class="mjx-chtml"><span aria-label="d" class="mjx-math"><span aria-hidden="true" class="mjx-mrow"><span class="mjx-mi"><span class="mjx-char MJXc-TeX-math-I" style="padding-top: 0.446em; padding-bottom: 0.298em; padding-right: 0.003em;">d</span></span></span></span></span></span></span>. </figcaption></figure>

Using this notation, a Markov Decision Process might look like

<figure class="image image_resized" style="width:78.77%"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/7a1eb02d34b965dd15356a3eaf4725d35c60c0b6fc871c06.png"/><figcaption>A Markov Decision Process, in the language of States, Actions, Rewards, and a policy <span><span class="mjpage"><span class="mjx-chtml"><span aria-label="\Pi" class="mjx-math"><span aria-hidden="true" class="mjx-mrow"><span class="mjx-mi"><span class="mjx-char MJXc-TeX-main-R" style="padding-top: 0.372em; padding-bottom: 0.372em;">Π</span></span></span></span></span></span></span>. Taken from the original article.</figcaption></figure>

### Response incentive, adversarial policy/state incentives

Finally, since we are interested in how the agent responds to \\(\\mathcal{P}_A\\), the last definitions the article introduces are state and policy adversarial incentives on \\(\\mathcal{P}_A\\), variations of the **Response Incentive** introduced in [a previous article](https://arxiv.org/abs/2102.01685). Response incentive is exemplified in this figure:

<figure class="image image_resized" style="width:86.97%"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/6ef0ff868511b5f34953e23f4f659bab119e6d1873f74878.png"/><figcaption>The professor has a Response Incentive with respect to the size of the Graduate class when deciding whether to upload the lecture online because it has both "an incentive and a means" to control that variable (using the words from <a href="https://arxiv.org/abs/2001.07118">Incentives that Shape Behavior</a>, from where the image is obtained). One of them is not present in Paper review or Student illness, so they lack the same response incentive.</figcaption></figure>

The response incentive will be called an *adversarial policy incentive* if the intersection between the "control path" going through \\(\\Pi\\) and the "information path" occurs before a state is reached by the former. Otherwise, it is called an *adversarial state incentive*.

<figure class="image image_resized" style="width:69.86%"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/d3213548934922815d551ccbfe6ab67364038bbbe7944fdb.png"/><figcaption>Causal Influence Diagram for a Modified Action MDP. The teal paths represent an adversarial policy incentive and the light pink an adversarial state incentive. Also taken from the original article.</figcaption></figure>

Reinforcement Learning Objectives
---------------------------------

### Reward maximization

Using these definitions we can explore how to generalize different objectives that appear in the Reinforcement Learning literature, from MDPs to MAMDPs. One simple alternative is the **reward maximization objective**:

<figure class="image image_resized" style="width:58.98%"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/4488af7820260393cf43051a600940464166c0695e2a6db3.png"/></figure>

### Bellman optimality objective

The reward maximization objective is perhaps the simplest objective, as it ignores the structure of the environment and just optimizes globally. This structureless optimization, however, may not always be the most efficient one. The most famous alternative is the **Bellman optimality objective**, which in its action-value form (see [Sutton and Barto, Reinforcement Learning](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) equation 3.20) says 

<figure class="image image_resized" style="width:70.33%"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/182c6892e1ad7d07935c81c58c7070fab6ca7bcc3c18c479.png"/></figure>

with \\(Q^{BO}\\) representing the action-value function, which scores how good is taking each action at each state. 

From this objective, Langlois and Everitt prove one of the most remarkable results from their article (Proposition 2): since \\(\\mathcal{P}_A\\) is not represented anywhere in those equations, an optimal policy for \\(\\mathcal{M}\\) will also be optimal for \\(\\tilde{\\mathcal{M}}\\). In other words, **an algorithm that optimizes the Bellman objective will ignore modifications given by**\\(\\mathcal{P}_A\\)**!**

### Virtual and empirical policy value objectives

Finally, the third objective studied in the article is that given by a greedy policy optimizing \\(Q\\) in the Bellman action-value equationfor the MDP \\(\\mathcal{M}\\), that we call the **policy value objective**:

<figure class="image image_resized" style="width:69.69%"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/e4928b3551622054f805446ecaa3dacf065e0ad526365365.png"/></figure>

The difference with the Bellman optimality value objective is that in that one we took \\(\\max_{a'\\in\\mathcal{A}}\\), while in this we take \\(\\mathbb{E}_{a'\\sim\\pi(s')}\\). The connection between both is that if the policy is greedy with respect to \\(Q\\), then the policy improvement theorem ([Sutton and Barto, Reinforcement Learning](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf), section 4.2) ensures that Bellman action value equation (4) converges to Bellman optimality objective (2) and (3).

Since in equation (4) the action is sampled from the policy in \\(\\mathcal{M}\\), there are two different ways of generalizing this equation to \\(\\tilde{\\mathcal{M}}\\), corresponding to whether the next action is still sampled from \\(\\pi\\) or from \\(\\mathcal{P}_A\\). 

The first is called **virtual policy value objective** and assumes that \\(a'\\) is still sampled from \\(\\pi(s')\\), while \\(\\pi\\) still is greedy.

<figure class="image image_resized" style="width:75.93%"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/e8d9c3892fd84532ab5a71676616602d69c48bb263adfa0f.png"/></figure>

Similarly to the Bellman optimality objective, there is no difference between \\(\\mathcal{M}\\) and \\(\\tilde{\\mathcal{M}}\\) between equations (4) and (5), as \\(\\mathcal{P}_A\\) is not represented. Consequently, the behavior of the agent in \\(\\tilde{\\mathcal{M}}\\) and the associated \\(\\mathcal{M}\\) are the same (Proposition 4). And since greedy policies \\(\\pi^{\\text{VPV}}\\) converge to the same behavior as if they were optimizing the Bellman optimality objective, **the virtual policy value objective ignores**\\(\\mathcal{P}_A\\)**, and converges to the Bellman optimality objective behavior***.*

The second alternative is called **empirical policy value objective** because it understands that \\(a'\\) will now be sampled from \\(\\mathcal{P}_A\\). 

<figure class="image image_resized" style="width:75.52%"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/c8b7b096cb574d1c35142831504675ae8762f4a25520b571.png"/></figure>

This objective is trickier because making the agent policy dependent on \\(\\mathcal{P}_A\\) may make it unstable: \\(\\pi^{\\text{EPV}}\\) depends on \\(\\mathcal{P}_A\\), which in turn depends on \\(\\pi^{\\text{EPV}}\\)[^fn0m5yz0i1gxu]. The behavior of an agent optimizing this objective is determined by the contrast between equations (7) and (8): In equation (8) \\(\\pi^{\\text{EPV}}\\) **does not take into account action modifications in the following step** because it assumes actions are determined by \\(\\pi^{\\text{EPV}}\\). As such it lacks an adversarial policy incentive. **However, action modifications by**\\(\\mathcal{P}_A\\)**are taken into account for any future actions** in equation (7), so it has an adversarial state incentive.

<figure class="image image_resized" style="width:67.78%"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/27f1f8f8ce1396492cd603e190e5290eb5035dfb7ac3c832.png"/><figcaption>(Partial) Causal Influence Diagram, with fixed <span><span class="mjpage"><span class="mjx-chtml"><span aria-label="\pi'" class="mjx-math"><span aria-hidden="true" class="mjx-mrow"><span class="mjx-msup"><span class="mjx-base" style="margin-right: -0.003em;"><span class="mjx-mi"><span class="mjx-char MJXc-TeX-math-I" style="padding-top: 0.225em; padding-bottom: 0.298em; padding-right: 0.003em;">π</span></span></span><span class="mjx-sup" style="font-size: 70.7%; vertical-align: 0.513em; padding-left: 0.076em; padding-right: 0.071em;"><span class="mjx-mo" style=""><span class="mjx-char MJXc-TeX-main-R" style="padding-top: 0.298em; padding-bottom: 0.298em;">′</span></span></span></span></span></span></span></span></span> in <span><span class="mjpage"><span class="mjx-chtml"><span aria-label="t=1" class="mjx-math"><span aria-hidden="true" class="mjx-mrow"><span class="mjx-mi"><span class="mjx-char MJXc-TeX-math-I" style="padding-top: 0.372em; padding-bottom: 0.298em;">t</span></span><span class="mjx-mo MJXc-space3"><span class="mjx-char MJXc-TeX-main-R" style="padding-top: 0.077em; padding-bottom: 0.298em;">=</span></span><span class="mjx-mn MJXc-space3"><span class="mjx-char MJXc-TeX-main-R" style="padding-top: 0.372em; padding-bottom: 0.372em;">1</span></span></span></span></span></span></span>, for the empirical value objective where one can see an adversarial policy incentive. The complete diagram would not be acyclic and therefore a Causal Influence Diagram. </figcaption></figure>

Temporal-difference algorithms
------------------------------

Using these objectives, we can study the behavior of different agents on \\(\\tilde{\\mathcal{M}}\\). In particular, we analyze Temporal-difference agents, defined by updates in the value function similar to

<figure class="image image_resized" style="width:57.51%"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/3e90f7152b30f69fbe6f1821dfb186f4a564facaba29167d.png"/></figure>

for \\(\\alpha\\) a small parameter.

### Q-Learning

The first agent the article analyses is Q-Learning, an off-policy agent:

<figure class="image image_resized" style="width:67.35%"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/8e3583e395d6aa47014b88e9a601308db78716be463aecac.png"/></figure>

Q-Learning is off-policy because it compares the current state and action pair against the maximum \\(Q(s_{t+1},a)\\), not with the action-value \\(Q\\) corresponding to the following action sampled by the policy. Consequently, it optimizes the Bellman optimality objective.

### Virtual and empirical SARSA

In contrast, SARSA is on-policy and compares the current action with the next one taken by the policy. therefore the two possible generalizations include taking into account the action sampled by the policy \\(\\pi\\), called virtual SARSA

<figure class="image image_resized" style="width:66.41%"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/99b35e55b54710b19306e9e8f29c75ee7caae20e74811394.png"/></figure>

or by \\(\\mathcal{P}_A\\), empirical SARSA

<figure class="image image_resized" style="width:66.51%"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/565870b26cf5ddb17382c90910b94d72035be0657cfa8c2d.png"/></figure>

Note that virtual and empirical SARSA optimize the virtual and optimal policy value objectives, respectively.

The article then proves the convergence of these algorithms as long as (i) they visit every state infinitely many times, which is however not possible to ensure in a MAMDP even if the agent is \\(\\varepsilon\\)-greedy; (ii) there exists a solution to (7) and (8) to which converge for empirical SARSA; and (iii) \\(\\Pi_t\\) converges to the greedy policy \\(\\Pi^* = \\arg\\max_aQ(s,a)\\) in both SARSA algorithms.

Experiments
-----------

### Simulation-oversight environment

Do these RL agents really exhibit this behavior when put into one MAMDP environment? One simple environment is the following: an agent has to first decide between working on a \\(\\texttt{Simulated}\\) environment and a \\(\\texttt{Real}\\) environment, and after that, between \\(\\texttt{Completing}\\)  a task as expected or \\(\\texttt{Exploiting}\\) a bug. The reward function is \\(+2\\) for choosing the \\(\\texttt{Real}\\) environment, and \\(+1\\) for \\(\\texttt{Exploiting}\\). However, if the agent decides to work on the \\(\\texttt{Real}\\) environment, a human will first check the behavior in the simulated environment, and if the agent \\(\\texttt{Exploits}\\) there, the reward will instead be \\(-2\\). Let us see how our agents behave:

<figure class="image image_resized" style="width:58.75%"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/3ee9a5b9602247f3d943e18b44bc218034ce3d10d5ce3583.png"/></figure>

Here the UH-CMA-ES is a reward maximization objective agent. This agent will learn to [deceive the human](https://www.alignmentforum.org/posts/zthDPAjh9w6Ytbeks/deceptive-alignment) supervisor by \\(\\texttt{Completing}\\) when situated on the \\(\\texttt{Simulated}\\) environment, but choosing the \\(\\texttt{Real}\\) environment and \\(\\texttt{Exploiting}\\) there, achieving a reward of \\(+3\\). In contrast, we can see that both Virtual SARSA and Q-Learning behave as if there was no supervisor \\(\\mathcal{P}_A\\).

The most complex case is Empirical SARSA. Understanding that it will ignore the supervision on the second time step, it avoids getting caught \\(\\texttt{Real}\\)-\\(\\texttt{Exploiting}\\) by choosing the \\(\\texttt{Simulated}\\) environment instead, and \\(\\texttt{Exploiting}\\) there.

### Off-switch and whisky gold

Agents were also tested in two (slightly modified) environments from the [AI Safety Gridworlds](https://arxiv.org/abs/1711.09883). The off-switch environment is an environment where a human may prevent the agent from reaching its goal if the agent does not disable the off-switch previously. On the other hand, the whisky-gold agent may take the action of drinking whisky which gives a small reward but makes its actions much more random and therefore complicates achieving the goal too.

The interesting aspect to notice in these environments is that those agents that correctly do not disable the off-switch, incorrectly drink the whisky, and vice-versa. This is because Q-Learning and Virtual SARSA do not take into account possible modifications in their policy or objectives, while Empirical SARSA and reward function maximization objective algorithms do. 

<figure class="image image_resized" style="width:49.94%"><img src="https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/e118f9be48c25234292801cdef7b898f04482ffc05c44ce1.png"/></figure>

The article also makes an interesting point: whether these agents will learn the action of \\(\\mathcal{P}_A\\) will depend on whether they are considered part of the state transition dynamics, or external to those. This boils down to whether the agent optimization procedure takes into account the action it would have taken, or that selected by \\(\\mathcal{P}_A\\).

Discussion
==========

One of the points discussed in the article is whether it is possible to encode the MAMDP in an MDP, eg, take \\(\\mathcal{P}_A\\) as part of  \\(\\mathcal{P}_S\\). The answer depends on whether \\(\\mathcal{P}_A(\\pi, s)\\) is only a function of the state, \\(\\mathcal{P}_A(s)\\). If such is the case, and the agent updates the \\(Q\\) function on his own action \\(a\\sim \\pi(s)\\) instead of the action taken by \\(a\\sim\\mathcal{P}_A(s)\\), then we recover an MDP with the same dynamics as the initial MADP. 

Another interesting point to see is the connection of these agents, in particular, Q-Learning and Virtual SARSA, with *time-unaware current reward function agents* discussed in the literature (see [Reward Tampering Problems and Solutions in Reinforcement Learning: A Causal Influence Diagram Perspective](https://arxiv.org/abs/1908.04734)). These agents and those seem to be connected in that they both ignore future modifications, but the modifications are different. The modifications of \\(\\mathcal{P}_{A}\\) only modify the next action. In contrast, the time-unaware current reward function agents were defined to study behavior under policy modifications. For this reason, this article works with slightly more restricted settings but is still rich enough to observe a rich set of behaviors. In particular, they allow us to observe the behavior of embedded agents.

Finally, I also find very interesting this article for a reason: in private conversations, I have often heard that the main limitation of causal incentive research is that there is really no causal diagram in the agent minds which we can analyze, or potentially even design our agent over. This is an important limitation and in fact, the main reason why I placed Causal Representation Learning in a central position in [my own research agenda](https://www.alignmentforum.org/posts/5BkEoJFEqQEWy9GcL/an-open-philanthropy-grant-proposal-causal-representation), I thought that without a causal representation of the environment causal analysis would not be possible, or be severely limited. This article is special because it shows otherwise, that there are cases in which we can predict or design the agent behavior just from the training algorithm even if there is no causal diagram over which to reason about. 

Some references
===============

[1] Eric Langlois and Tom Everitt, [How RL Agents Behave When Their Actions Are Modified?](https://www.aaai.org/AAAI21Papers/AAAI-5767.LangloisE.pdf)

[2] Tom Everitt, Ryan Carey, Eric Langlois, Pedro A Ortega, Shane Legg, [Agent Incentives: A Causal Perspective](https://arxiv.org/abs/2102.01685).

[3] Tom Everitt, Marcus Hutter, Ramana Kumar, Victoria Krakovna, [Reward Tampering Problems and Solutions in Reinforcement Learning: A Causal Influence Diagram Perspective](https://arxiv.org/abs/1908.04734).

[4] Tom Everitt, Pedro A. Ortega, Elizabeth Barnes, Shane Legg, [Understanding Agent Incentives using Causal Influence Diagrams. Part I: Single Action Settings.](https://arxiv.org/abs/1902.09980)

[5] Marius Hobban, [Causality, Transformative AI and alignment - part I](https://www.alignmentforum.org/posts/oqzasmQ9Lye45QDMZ/causality-transformative-ai-and-alignment-part-i).

[6] Pablo A M Casares, [An Open Philanthropy grant proposal: Causal representation learning of human preferences](https://www.lesswrong.com/posts/5BkEoJFEqQEWy9GcL/an-open-philanthropy-grant-proposal-causal-representation).

[7] Jan Leike, Miljan Martic, Victoria Krakovna, Pedro A. Ortega, Tom Everitt, Andrew Lefrancq, Laurent Orseau, Shane Legg [AI Safety Gridworlds](https://arxiv.org/abs/1711.09883).

---

[^fn0m5yz0i1gxu]: The article does not provide a characterization of under which situation this self-referential behavior is stable. It is an interesting question worth addressing in the future.




