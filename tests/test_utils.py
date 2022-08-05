import numpy as np

from simplerl import discount_rewards, compare_lists_of_np_arrays, unzip_experience_buffer, BasicExperienceBuffer

class TestUtils:
    def test_discounted_reward_computation(self):
        gamma = 0.9
        assert np.array_equal(discount_rewards(np.array([1.0, 0.0, 2.0]), gamma), np.array([1 + pow(gamma, 2)*2, gamma*2, 2]))

    def test_compare_lists_of_np_arrays(self):
        l1 = [np.array([1,2]), np.array([2,3])]
        l2 = [np.array([1,2]), np.array([2,4])]
        assert compare_lists_of_np_arrays(l1, l2) == False
        assert compare_lists_of_np_arrays(l2, l1) == False
        assert compare_lists_of_np_arrays(l1, l1) == True
        assert compare_lists_of_np_arrays(l2, l2) == True

    def test_unzip_experience_buffer(self):
        buffer = BasicExperienceBuffer(size=10, batch_size=2)
        buffer.add(np.array([1, 8]), 2, 3, np.array([4, 8]))
        buffer.add(np.array([5, 8]), 6, 7, np.array([8, 8]))
        buffer.add(np.array([9, 8]), 10, 11, np.array([12, 8]))

        states, actions, rewards, next_states = unzip_experience_buffer(buffer)
        assert compare_lists_of_np_arrays(states, [np.array([1, 8]), np.array([5, 8]), np.array([9, 8])])
        assert compare_lists_of_np_arrays(actions, [2, 6, 10])
        assert compare_lists_of_np_arrays(rewards, [3, 7, 11])
        assert compare_lists_of_np_arrays(next_states, [np.array([4, 8]), np.array([8, 8]), np.array([12, 8])])