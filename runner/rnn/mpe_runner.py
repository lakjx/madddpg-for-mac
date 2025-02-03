import numpy as np
import torch
import time

from runner.rnn.base_runner import RecRunner

class MPERunner(RecRunner):
    """Runner class for Multiagent Particle Envs (MPE). See parent class for more information."""
    def __init__(self, config, test_mode=False):
        super(MPERunner, self).__init__(config, test_mode)
        self.collecter = self.shared_collect_rollout if self.share_policy else self.separated_collect_rollout
        # fill replay buffer with random actions
        # fill replay buffer with random actions
        num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
        self.warmup(num_warmup_episodes) if not test_mode else None
        self.start = time.time()
        self.log_clear()
    
    def eval(self):
        """Collect episodes to evaluate the policy."""
        self.trainer.prep_rollout()
        eval_infos = {
        'average_episode_rewards': [],
        'Goodput': [],
        'Packet_Received_Ratio': [],
        'Collision_Ratio': [],
        'buffer_occupancy': []
        }
        
        for _ in range(self.args.num_eval_episodes):
            env_info = self.collecter( explore=False, training_episode=False, warmup=False)
            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")
    
    @torch.no_grad()
    def test(self,num_episodes=1):
        """Collect episodes to test the policy."""
        self.trainer.prep_rollout()
        dataset = []
        for _ in range(1):
            env_info = self.collecter( explore=False, training_episode=False, warmup=False, test=True)
            #从episode_records中获取数据
            episode_data = []
            for i in range(len(self.episode_records['state'])):
                data_point = {
                    "observation":{
                        "step": self.episode_records['state'][i]['step'],
                        "agents_obs": self.episode_records['state'][i]['observation'],
                        #more imformation
                        "env_metrics": {
                            "Goodput": env_info['Goodput'],
                            "Packet_Received_Ratio": env_info['Packet_Received_Ratio'],
                            "Collision_Ratio": env_info['Collision_Ratio'],
                            "buffer_occupancy": env_info['buffer_occupancy']
                        }
                    },
                    "action":{
                        "step": self.episode_records['action'][i]['step'],
                        "agents_actions": self.episode_records['action'][i]['action']
                    }
                }
                episode_data.append(data_point)
            dataset.extend(episode_data)

        format_dat = self.format_for_llm(dataset)
        self.save_dataset(format_dat, "marl_training_data.jsonl")
    
    def save_dataset(self,dataset,filename):
        import json
        import os
        with open(filename, 'w', encoding='utf-8') as f:
            for item in dataset:
                json_str = json.dumps(item, ensure_ascii=False)
                f.write(json_str + '\n')

    def format_for_llm(self,dataset):
        """将数据格式化为适合LLM训练的格式"""
        formatted_data = []
        for item in dataset:
            prompt = f"""
            Environment State:
            Step: {item['observation']['step']}
            Agents Observations: {item['observation']['agents_obs']}
            Environment Metrics:
            - Goodput: {item['observation']['env_metrics']['Goodput']}
            - Packet Received Ratio: {item['observation']['env_metrics']['Packet_Received_Ratio']}
            - Collision Ratio: {item['observation']['env_metrics']['Collision_Ratio']}
            - Buffer Occupancy: {item['observation']['env_metrics']['buffer_occupancy']}
            
            Based on the above state, what actions should the agents take?
            """
            
            completion = f"""
            The agents should take the following actions:
            {item['action']['agents_actions']}
            """
            
            formatted_data.append({
                "prompt": prompt,
                "completion": completion
            })
        
        return formatted_data

    # for mpe-simple_spread and mpe-simple_reference  
    @torch.no_grad() 
    def shared_collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Collect a rollout and store it in the buffer. All agents share a single policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        # only 1 policy since all agents share weights
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if training_episode or warmup else self.eval_env
        env.envs[0].is_training = training_episode or warmup
        obs = env.reset() #shape (1,agent_num,obs_dim)

        rnn_states_batch = np.zeros((self.num_envs * self.num_agents, self.hidden_size), dtype=np.float32)
        last_acts_batch = np.zeros((self.num_envs * self.num_agents, policy.output_dim), dtype=np.float32)

        # initialize variables to store episode information.
        episode_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_acts = {p_id : np.zeros((self.episode_length, self.num_envs, self.num_agents, policy.output_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id : np.zeros((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id : np.ones((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id : np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_avail_acts = {p_id : None for p_id in self.policy_ids}

        t = 0
        while t < self.episode_length:
            share_obs = obs.reshape(self.num_envs, -1)
            # group observations from parallel envs into one batch to process at once
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env
            if warmup:
                # completely random actions in pre-training warmup phase
                acts_batch = policy.get_random_actions(obs_batch)
                # get new rnn hidden state
                _, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                            last_acts_batch,
                                                            rnn_states_batch)
            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                acts_batch, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                                    last_acts_batch,
                                                                    rnn_states_batch,
                                                                    t_env=self.total_env_steps,
                                                                    explore=explore)
            acts_batch = acts_batch if isinstance(acts_batch, np.ndarray) else acts_batch.cpu().detach().numpy()
            # update rnn hidden state
            rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            last_acts_batch = acts_batch

            env_acts = np.split(acts_batch, self.num_envs)
            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)
            if training_episode:
                self.total_env_steps += self.num_envs

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or t == self.episode_length - 1

            episode_obs[p_id][t] = obs
            episode_share_obs[p_id][t] = share_obs
            episode_acts[p_id][t] = np.stack(env_acts)
            episode_rewards[p_id][t] = rewards
            episode_dones[p_id][t] = dones
            episode_dones_env[p_id][t] = dones_env
            t += 1

            obs = next_obs

            if terminate_episodes:
                break

        episode_obs[p_id][t] = obs
        episode_share_obs[p_id][t] = obs.reshape(self.num_envs, -1)

        if explore:
            self.num_episodes_collected += self.num_envs
            # push all episodes collected in this rollout step to the buffer
            self.buffer.insert(self.num_envs,
                               episode_obs,
                               episode_share_obs,
                               episode_acts,
                               episode_rewards,
                               episode_dones,
                               episode_dones_env,
                               episode_avail_acts)

        average_episode_rewards = np.mean(np.sum(episode_rewards[p_id], axis=0))
        env_info['average_episode_rewards'] = average_episode_rewards

        return env_info

    # for mpe-simple_speaker_listener
    @torch.no_grad()
    def separated_collect_rollout(self, explore=True, training_episode=True, warmup=False, test=False):
        """
        Collect a rollout and store it in the buffer. Each agent has its own policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        if test:
            episode_records = {
                'state': [],
                'action': [],
            }

        env_info = {}
        env = self.env if training_episode or warmup else self.eval_env
        env.envs[0].is_training = training_episode or warmup

        obs = env.reset()

        rnn_states = np.zeros((self.num_agents, self.num_envs, self.hidden_size), dtype=np.float32)

        last_acts = {p_id : np.zeros((self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].output_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_acts = {p_id : np.zeros((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].output_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id : np.zeros((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id : np.ones((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id : np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_avail_acts = {p_id : None for p_id in self.policy_ids}

        t = 0
        while t < self.episode_length:
            if test:
                current_state = {
                    'step': t,
                    'observation': [oo.tolist() if isinstance(oo, np.ndarray) else oo for oo in obs[0]],
                }
                episode_records['state'].append(current_state)

            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                policy = self.policies[p_id]
                agent_obs = np.array([each_env_obs[agent_id] for each_env_obs in obs])  # agent_obs = np.stack(obs[:, agent_id])
                # share_obs = np.concatenate([obs[0, i] for i in range(self.num_agents)]).reshape(self.num_envs,
                #                                                                                 -1).astype(np.float32)
                share_obs = np.concatenate([obs[0][i] for i in range(self.num_agents)]).reshape(self.num_envs,-1).astype(np.float32)
                # get actions for all agents to step the env
                if warmup:
                    # completely random actions in pre-training warmup phase
                    # [parallel envs, agents, dim]
                    act = policy.get_random_actions(agent_obs)
                    # get new rnn hidden state
                    _, rnn_state, _ = policy.get_actions(agent_obs,
                                                        last_acts[p_id][:, 0],
                                                        rnn_states[agent_id])
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    if self.algorithm_name == "rmasac":
                        act, rnn_state, _ = policy.get_actions(agent_obs,
                                                                last_acts[p_id],
                                                                rnn_states[agent_id],
                                                                sample=explore)
                    else:
                        act, rnn_state, _ = policy.get_actions(agent_obs,
                                                                last_acts[p_id].squeeze(axis=0),
                                                                rnn_states[agent_id],
                                                                t_env=self.total_env_steps,
                                                                explore=explore)
                # update rnn hidden state
                rnn_states[agent_id] = rnn_state if isinstance(rnn_state, np.ndarray) else rnn_state.cpu().detach().numpy()
                last_acts[p_id] = np.expand_dims(act, axis=1) if isinstance(act, np.ndarray) else np.expand_dims(act.cpu().detach().numpy(), axis=1)

                episode_obs[p_id][t] = agent_obs.reshape(self.num_envs,1,-1)
                episode_share_obs[p_id][t] = share_obs.reshape(self.num_envs,1,-1)
                episode_acts[p_id][t] = act.reshape(self.num_envs,1,-1) if isinstance(act, np.ndarray) else act.cpu().detach().numpy().reshape(self.num_envs,1,-1)



            env_acts = []
            for i in range(self.num_envs):
                env_act = []
                for p_id in self.policy_ids:
                    p_dim = self.policy_act_dim[p_id][0] if isinstance(self.policy_act_dim[p_id],np.ndarray) else self.policy_act_dim[p_id]
                    if p_dim == 0:
                        env_act.append(np.array([]))
                        continue
                    else:
                        d_act1 = last_acts[p_id][i, 0][:p_dim].argmax()
                        if last_acts[p_id][i, 0][p_dim:].size > 0:
                            d_act2 = last_acts[p_id][i, 0][p_dim:].argmax() # communication action
                            env_act.append(np.array([d_act1, d_act2]))
                        else:
                            env_act.append(np.array([d_act1]))
                        # env_act.append(last_acts[p_id][i, 0])
                env_acts.append(env_act)
            if test:
                current_action = {
                    'step': t,
                    'action': [aa.tolist() if isinstance(aa, np.ndarray) else aa for aa in env_acts[0]],
                }
                episode_records['action'].append(current_action)
            
            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or t == self.episode_length - 1
            if terminate_episodes:
                dones_env = np.ones_like(dones_env).astype(bool)


            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                # episode_rewards[p_id][t] = np.expand_dims(rewards[:, agent_id], axis=1)
                # episode_dones[p_id][t] = np.expand_dims(dones[:, agent_id], axis=1)
                # episode_dones_env[p_id][t] = dones_env
                episode_rewards[p_id][t] = np.expand_dims(np.array([env_rewards[agent_id] for env_rewards in rewards]), axis=1)
                episode_dones[p_id][t] = np.expand_dims(np.array([env_dones[agent_id] for env_dones in dones]), axis=1)
                episode_dones_env[p_id][t] = dones_env

            obs = next_obs
            t += 1

            if training_episode:
                self.total_env_steps += self.num_envs

            if terminate_episodes:
                break
        if test:
            self.episode_records = episode_records
        #     import json
        #     import os
        #     save_dir = self.args.test_record_dir
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
            #标记用户数量的到达率p_SDU_arrival
            # logo = f'UE={self.args.UE_num}_p={self.args.p_SDU_arrival}'
            # file_name = os.path.join(save_dir, logo + ".json")
            # try:
            #     with open(file_name, 'w') as f:
            #         json.dump(episode_records, f, indent=4,separators=(',', ': '))
            #     print(f"JSON file has been created at {file_name}")
            # except Exception as e:
            #     print(f"Failed to create JSON file: {e}")

        for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
            # episode_obs[p_id][t] = np.stack(obs[:, agent_id])
            # episode_share_obs[p_id][t] = np.concatenate([obs[0, i] for i in range(self.num_agents)]).reshape(self.num_envs,
            #                                                                                     -1).astype(np.float32)
            agent_obs = np.array([each_env_obs[agent_id] for each_env_obs in obs])
            episode_share_obs[p_id][t] = np.concatenate([obs[0][i] for i in range(self.num_agents)]).reshape(self.num_envs, -1).astype(np.float32)


        if explore:
            self.num_episodes_collected += self.num_envs
            self.buffer.insert(self.num_envs, episode_obs, episode_share_obs, episode_acts, episode_rewards, episode_dones, episode_dones_env, episode_avail_acts)

        average_episode_rewards = []
        for p_id in self.policy_ids:
            average_episode_rewards.append(np.mean(np.sum(episode_rewards[p_id], axis=0)))

        env_info['average_episode_rewards'] = np.mean(average_episode_rewards)
        env_info['Goodput'] = env.envs[0].get_Goodput()
        env_info['Packet_Received_Ratio'] = env.envs[0].get_packet_arrival_rate()
        env_info['Collision_Ratio'] = env.envs[0].get_collision_rate()
        env_info['buffer_occupancy'] = np.average(env.envs[0].get_buffer_occupancy())

        return env_info

    def log(self):
        """See parent class."""
        end = time.time()
        print("\n Env {} Algo {} Exp {} runs total num timesteps {}/{}, FPS {}.\n"
              .format(self.args.scenario_name,
                      self.algorithm_name,
                      self.args.experiment_name,
                      self.total_env_steps,
                      self.num_env_steps,
                      int(self.total_env_steps / (end - self.start))))
        for p_id, train_info in zip(self.policy_ids, self.train_infos):
            self.log_train(p_id, train_info)

        self.log_env(self.env_infos)
        self.log_clear()

    def log_clear(self):
        """See parent class."""
        self.env_infos = {}

        self.env_infos['average_episode_rewards'] = []
        self.env_infos['Goodput'] = []
        self.env_infos['Packet_Received_Ratio'] = []
        self.env_infos['Collision_Ratio'] = []
        self.env_infos['buffer_occupancy'] = []
