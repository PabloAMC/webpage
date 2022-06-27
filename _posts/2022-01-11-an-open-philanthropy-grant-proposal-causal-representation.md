---
layout: post
title: "An Open Philanthropy grant proposal: Causal representation learning of human preferences"
date: 2022-01-11 11:28:01 UTC
#subtitle: 
#background:
---

*This is a proposal I wrote for the recent [Open Philanthropy call on AI Alignment projects](https://www.openphilanthropy.org/focus/global-catastrophic-risks/potential-risks-advanced-artificial-intelligence/request-for-proposals-for-projects-in-ai-alignment-that-work-with-deep-learning-systems). I am finishing my Ph.D. in quantum computing and I would like to start working on AI Alignment. Thus, this is my proposal for a postdoc-style grant to learn more about one area I think the community could pay more attention to. However, since I am quite new to this area, there might be things that I have overlooked. For that reason, if you see something that doesn't sound promising, or you disagree with it, I would be happy to know about it.*


*The proposal received some light comments from Victor Veitch, Ryan Carey, and José Hernández-Orallo. I also received some comments from Jaime Sevilla during the ideation phase. I am really grateful to all of them for their help. Any remaining errors and inaccuracies are solely my fault.*


Summary: Causal Representation Learning aims to extract a causal and disentangled representation of the mechanisms in the environment. I propose a postdoctoral project studying its applicability to learning human preferences, which should make the learned model robust and interpretable. In other words, human preferences have a causal structure, can we learn its concepts and their causal relations?


Introduction
============


One of the most promising and straightforward research lines in AI Safety is what is known as value learning (Russell, 2019): developing ways for the system to infer our preferences to subsequently optimize for them. While a good initial step, there are two main limitations with this approach: (1) inferring these preferences is often very complex, and (2) it only addresses the problem of outer alignment, that of aligning the optimizer with the user. In contrast, inner alignment can be seen as a robustness problem: the optimized model might have learned a representation of the objective that is correlated with the true objective but breaks on out-of-distribution situations. Consequently, to solve both problems, we need a way to robustly infer human preferences that extends to such cases.


Causal Representation Learning (CLR) offers a possible solution: the key objective of CLR is to learn the causal structure of the environment. Using this tool might enable AI systems to learn causal representations of what people want, and could as a result generalize better and be more interpretable. Furthermore, it may enable the application of causal incentive analysis and design as studied in the <causalincentives.org> research group.


What is Causal Representation Learning?
=======================================


Causal Representation Learning is the “discovery of high-level causal variables from low-level observations” (Schölkopf, 2021). The main idea is to be able to reconstruct causal models of the world, usually in the form of Structural Causal Models, and embed them in modern neural ML systems. Doing so addresses some of the main limitations of current machine learning models:


* The current state-of-the-art models trained under independent and identically distributed (iid) data assumptions are not robust to distributional changes (Koh, 2021).
* Such models might also be narrow because they are not guaranteed to learn reusable mechanisms that they can apply to new environments. However, there are some indications that large models do learn disentangled representations in some cases (Besserve, 2018).
* Finally, the need for iid data makes them require a lot of training data. Furthermore, since there is no need to learn the individual causal components of the world in a disentangled manner, they could be less transparent.


By providing a causal perspective, one aims to enable such systems to learn the independent (disentangled) causal mechanisms of the world. Moreover, this approach seems especially well suited to Reinforcement Learning settings, where the interactive nature facilitates learning causal relations instead of only observational (Bareimboim, 2020).


For these reasons, there is in my opinion a reasonable chance (at least 25% but with a lot of uncertainty) that these techniques could become an important component of future agentic general intelligent systems, capable of adapting to unseen situations. If such systems use these techniques, I hope it might be easy to repurpose them for value learning. Additionally, I believe human preferences and values are likely to possess some causal structure. Indeed, the idea of inferring human values from preferences can be framed as a causal inference problem although in the literature causal inference usually assumes the causal model is known. Consequently, it seems natural and better to attempt using causal representation learning techniques (as opposed to simpler statistical learning) of human preferences. My proposal aims to explore if this is indeed feasible, and what are the limitations.


There are however some challenges that could hinder the effectiveness of these techniques. The main one is that even when provided with a causal model, causal inference is generally an expensive procedure that quickly becomes intractable. Nevertheless, there are some neural models that can partially address this limitation, at least to the interventional layer (Zečević, 2021; Zečević, 2021a), and since this is a rather new research area, it might be possible to discover practical methods in particular situations. These models are key because we generally prefer flexible and scalable methods to hand-crafted ones that rely on expert world knowledge. While causal modeling is often associated with the latter, neural causal models non-parametrically handle causal structure in a way that improves robustness on extrapolation without sacrificing the scaling performance required by “the bitter lesson” (Sutton, 2019). Furthermore, my intuition then is that CRL is not aimed to fully substitute current non-causal methods. Rather, when a new situation is found the system will use it (with the corresponding computational expense), and otherwise will default to other more standard and not so expensive methods.


The second criticism is that current CRL methods still do not achieve state-of-the-art performance on most supervised ML settings (Besserve, 2018). A particular good overview of the limitations of causal inference techniques and causal discovery (related but not the same as causal representation learning) can be found in (Sevilla, 2021). On the other hand, prominent researchers such as Yoshua Bengio or Bernhard Scholkopf are betting on this field, so I am not sure how important it will be for the future of AI. In any case, I believe there has not been any research carried out on how to use these techniques for value learning purposes, and it may be worth trying because it is a rather intuitive approach to tell apart the true things people care about from those that are merely correlated. Importantly, this confusion is what gives rise to one of the best-known problems in AI Safety, Goodhart’s law (Goodhart, 1975).


What we could gain?
===================


The key reason for this proposal is that if we want value learning to be a feasible AI alignment strategy, we need to make it as robust as possible to the changes the AI itself will produce (and any other). Consequently, causally representing what humans care about is probably a necessary condition to attain such robustness, as non-causal representations are almost certainly doomed to fail in out-of-distribution conditions. As an example, some research has already explored causal imitation learning to address its robustness vs distributional shift due to causal misidentification (de Haan, 2019; Zhang 2020).


I believe that while CRL as a research area is fairly new, there is already enough theory to explore how a causal version of value learning might look like. Then, if causal representation learning techniques become mainstream in advanced AI systems, we could routinely apply them to obtain more robust versions of human preferences. The ultimate robustness achieved will depend on both how good those techniques become at finding the independent mechanisms and concepts in our preferences, and how consistent are those. In fact, a strong assumption of this proposal is that humans’ theory of mind can be approximated with causal models, perhaps taking into account some psychological bias. However, I would like to highlight here that while this is a relatively large assumption, I think there is value in trying to see how far can we go with this educated guess. Finding out that CRL is a good way to learn efficient representations of how the world works but not of human preferences could be quite worrying because it would mean that one important tool for AI Alignment (value learning) is most likely doomed to fail, in my opinion.


Second, disentangled representations are much easier to interpret because they could correspond to concepts similar to those used in natural language, and their neural interpretation could be made modularly. As such, providing feedback to these systems could become easier because we could ideally identify the differences between our models and preferences, and what they have understood. Further, such causal representations can be done both for RL behavior and natural language, so one could even aim to obtain systems capable of explaining their beliefs in natural language.


Third, this research line could help design and implement agents with the proper causal incentives, or it could at least let us analyze their instrumental objectives using the techniques developed already by the causal incentives group ([causalincentives.com](http://causalincentives.com)).


Proposed steps
==============


I am currently finishing my Ph.D. in Physics (in quantum algorithms) so I would like to use this project as the foundation for a postdoc and a career transition, carried out under the supervision of Victor Veitch at the University of Chicago. As such, I will propose a series of research projects that I consider worthy of carrying out, that if successful, could each constitute a publication.


As mentioned previously, one problem I consider to be especially straightforward and useful is in the application of Causal Representation Learning of Human preferences. I believe the reason why this is feasible is that in various versions of Inverse Reinforcement Learning one aims to learn a representation of human preferences. For example, we could aim to carry out experiments similar to those in (Christiano, 2017) in Atari games but using causal representation learning techniques described in the literature, and compare the performance with the original paper. Then, we could evaluate how well it behaves in modified environments, and whether it is able to efficiently reuse what it learned in previous environments to quickly adapt to modified objectives provided by human feedback. If this works out well, we could aim to explore the limitations of this technique to address a variety of problems, including situations where inner alignment has been shown to be a problem.


It would also be interesting to know how we can use active learning-like or other kinds of interactive techniques to effectively detect and disentangle features even when they are correlated in human training data or behavior. This could be a second and related project.


A third one might aim to extract causal representations of human language (Wang, 2021; Zhijin) to enable AI systems to learn from human language, a complementary approach to using RL-based techniques. A possible question could be: “can we extract causal representations from both behavior and language and match them to enable systems to explain in plain language their behavior?” A source of data could be descriptions of people playing video games. The ultimate objective of this project would be training a system to explain in natural language why one player is taking some actions and not others. A challenge here is that unsupervised learning alone is often not sufficient to learn such disentangled representations (Locatello, 2019), but weak supervision is (Locatello, 2020). However, this is probably the most challenging idea to implement.


Finally, a fourth project might attempt to devise ways to design and train agents to have appropriate incentive structures according to the feedback tampering problem one aims to address (Everitt, 2021; Everitt, 2021a).


Conclusion
==========


Causal Reinforcement Learning aims to enable AI systems to learn concepts and their relationships in a similar way to how one would expect humans to do so: using causality. While I think that such a causality approach might need to be complemented with psychology insights to be useful, I think it is a natural extension of value learning research and it can be particularly useful in situations different from where training data was extracted, which I expect to be the most problematic in AI Safety. This causal approach promises to provide robustness and interpretability to ML methods (at greater computational expense), and for this reason, it is in my opinion worth being explored as a (partial) solution to AI Safety. Since I am not aware of anyone trying to use these techniques in AI Safety, I propose a postdoc research project to explore how promising they are.


Bibliography
============


(Baremboim, 2020) Elias Bareinboim, Causal Reinforcement Learning, tutorial at ICML 2020 <https://crl.causalai.net/>.


(Besserve, 2018) Besserve, Michel, et al. "Counterfactuals uncover the modular structure of deep generative models." arXiv preprint arXiv:1812.03253 (2018).


(Christiano, 2017) Christiano, Paul, et al. "Deep reinforcement learning from human preferences." arXiv preprint arXiv:1706.03741 (2017).


(de Haan, 2019) de Haan, Pim, Dinesh Jayaraman, and Sergey Levine. "Causal confusion in imitation learning." Advances in Neural Information Processing Systems 32 (2019): 11698-11709.


(Everitt, 2021) Everitt, Tom, et al. "Reward tampering problems and solutions in reinforcement learning: A causal influence diagram perspective." Synthese (2021): 1-33.


(Everitt, 2021a) Everitt, Tom, et al. "Agent incentives: A causal perspective." Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence,(AAAI-21). Virtual. 2021.


(Goodhart, 1975) Charles E. Goodhart. Problems of Monetary Management: The U.K. Experience 1975. Papers in Monetary Economics. Reserve Bank of Australia.


(Koh, 2021) Koh, Pang Wei, et al. "Wilds: A benchmark of in-the-wild distribution shifts." International Conference on Machine Learning. PMLR, 2021.


(Locatello, 2019) Locatello, Francesco, et al. "Challenging common assumptions in the unsupervised learning of disentangled representations." international conference on machine learning. PMLR, 2019.


(Locatello, 2020) Locatello, Francesco, et al. "Disentangling factors of variation using few labels." arXiv preprint arXiv:1905.01258 In International Conference on Learning Representations (2020).


(Russell, 2019) Russell, Stuart. Human compatible: Artificial intelligence and the problem of control. Penguin, 2019.


(Schölkopf, 2021) Schölkopf, Bernhard, et al. "Toward causal representation learning." Proceedings of the IEEE 109.5 (2021): 612-634.


(Sevilla, 2021) Sevilla, Jaime. “The limits of graphical causal discovery”. <https://towardsdatascience.com/the-limits-of-graphical-causal-discovery-92d92aed54d6>


(Sutton, 2019) Sutton, Rich. “The Bitter Lesson” <http://www.incompleteideas.net/IncIdeas/BitterLesson.html>


(Wang, 2021) Wang, Yixin, and Michael I. Jordan. "Desiderata for representation learning: A causal perspective." arXiv preprint arXiv:2109.03795 (2021).


(Zečević, 2021) Zečević, Matej, Devendra Singh Dhami, and Kristian Kersting. "On the Tractability of Neural Causal Inference." arXiv preprint arXiv:2110.12052 (2021).


(Zečević, 2021a) Zečević, Matej, et al. "Relating Graph Neural Networks to Structural Causal Models." arXiv preprint arXiv:2109.04173 (2021).


(Zhang, 2020) Zhang, Junzhe, Daniel Kumor, and Elias Bareinboim. "Causal imitation learning with unobserved confounders." Advances in neural information processing systems 33 (2020).


(Zhijing) Zhijin Jin, Causality for Natural Language Processing, <https://githubmemory.com/repo/zhijing-jin/Causality4NLP_Papers>



