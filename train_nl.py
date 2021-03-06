from PokerRL.game.games import DiscretizedNLHoldem  # or any other game
from PokerRL.eval.rl_br.RLBRArgs import RLBRArgs
from PokerRL.eval.lbr.LBRArgs import LBRArgs

from PokerRL.game import bet_sets

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(name="NL2",
                                         nn_type="recurrent",  # We also support RNNs, but the paper uses FF
                                         agent_bet_set=bet_sets.POT_ONLY,
                                         local_crayon_server_docker_address="46.101.123.20",

                                         DISTRIBUTED=False,
                                         CLUSTER=False,
                                         n_learner_actor_workers=1,  # 20 workers

                                         # regulate exports
                                         export_each_net=False,
                                         checkpoint_freq=1,
                                         eval_agent_export_freq=1,  # produces around 15GB over 150 iterations!

                                         n_actions_traverser_samples=3,  # = external sampling in FHP
                                         n_traversals_per_iter=15000,
                                         n_batches_adv_training=4000,
                                         mini_batch_size_adv=512,  # *20=10240
                                         init_adv_model="random",

                                         use_pre_layers_adv=True,
                                         n_cards_state_units_adv=192,
                                         n_merge_and_table_layer_units_adv=64,
                                         n_units_final_adv=64,

                                         max_buffer_size_adv=2e6,  # *20 LAs = 40M
                                         lr_adv=0.001,
                                         lr_patience_adv=99999999,  # No lr decay

                                         n_batches_avrg_training=20000,
                                         mini_batch_size_avrg=1024,  # *20=20480
                                         init_avrg_model="random",

                                         use_pre_layers_avrg=True,
                                         n_cards_state_units_avrg=192,
                                         n_merge_and_table_layer_units_avrg=64,
                                         n_units_final_avrg=64,

                                         max_buffer_size_avrg=2e6,
                                         lr_avrg=0.001,
                                         lr_patience_avrg=99999999,  # No lr decay

                                         log_verbose=True,


                                         game_cls=DiscretizedNLHoldem,

                                         lbr_args=LBRArgs(
                                            lbr_bet_set=bet_sets.POT_ONLY,
                                            n_lbr_hands_per_seat=5000, #10x *2
                                            lbr_check_to_round=2, #TURN 
                                         ),
                                         # You can specify one or both modes. Choosing both is useful to compare them.
                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_SINGLE,  # SD-CFR
                                             #EvalAgentDeepCFR.EVAL_MODE_AVRG_NET,  # Deep CFR
                                         ),

                                         ),
                  eval_methods={
                      "lbr": 5,
                  },
		iteration_to_import=18,
		name_to_import="NL2_",
                  n_iterations=9999999)
    ctrl.run()
