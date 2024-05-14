import unittest
import tempfile
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aer_utils.utils import calculate_time_saer, calculate_word_saer


class TestAER(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_evaluate_on_target(self):
 
        path_gold_alignment = '/private/home/alastruey/speech_gold_alignment/dataset/alignment_no_punct.txt'
        metrics_w_saer = calculate_time_saer(path_gold_alignment, path_gold_alignment, 'de-en', 's2t')
        metrics_t_saer = calculate_word_saer(path_gold_alignment, path_gold_alignment, 'de-en')
 
        self.assertTrue(metrics_w_saer['precision'][0] == 1.0)
        self.assertTrue(metrics_w_saer['recall'][0] == 1.0)
        self.assertTrue(metrics_w_saer['aer'][0] == 0.0)

        self.assertTrue(metrics_t_saer['precision'][0] == 1.0)
        self.assertTrue(metrics_t_saer['recall'][0] == 1.0)
        self.assertTrue(metrics_t_saer['aer'][0] == 0.0)

    def test_s2t_ende(self):
        alignment_hyp = '1-2 2-1'
        alignment_gold = '1-2 2-1 3-3' # de-en always

        temp_align_path = os.path.join(self.temp_dir.name, 'align_hyp.txt')
        with open(temp_align_path, 'w') as f:
            f.write(alignment_hyp)
        
        temp_align_gold_path = os.path.join(self.temp_dir.name, 'align_gold.txt')
        with open(temp_align_gold_path, 'w') as f:
            f.write(alignment_gold)


        metrics = calculate_word_saer(temp_align_path, temp_align_gold_path, 'en-de')

        self.assertTrue(metrics['aer'][0] == 1.0-((2.0+2.0)/(2.0+3.0)))

    def test_w_saer_ende_deen(self):
        alignment_hyp_ende = '1-1 2-4 3-2 4-4'
        alignment_hyp_deen = '1-1 4-2 2-3 4-4 3-5'
        alignment_gold = '1-1 1-5 2-3 3-4 4-2' # de-en always

        temp_align_path_ende = os.path.join(self.temp_dir.name, 'align_hyp_ende.txt')
        with open(temp_align_path_ende, 'w') as f:
            f.write(alignment_hyp_ende)

        temp_align_path_deen = os.path.join(self.temp_dir.name, 'align_hyp_deen.txt')
        with open(temp_align_path_deen, 'w') as f:
            f.write(alignment_hyp_deen)
        
        temp_align_gold_path = os.path.join(self.temp_dir.name, 'align_gold.txt')
        with open(temp_align_gold_path, 'w') as f:
            f.write(alignment_gold)


        metrics_ende = calculate_word_saer(temp_align_path_ende, temp_align_gold_path, 'en-de')
        metrics_deen = calculate_word_saer(temp_align_path_deen, temp_align_gold_path, 'de-en')

        self.assertTrue(metrics_ende['aer'][0] == 1.0-((3.0+3.0)/(4.0+5.0)))
        self.assertTrue(metrics_deen['aer'][0] == 1.0-((3.0+3.0)/(5.0+5.0)))

    def test_p_align(self):
        alignment_hyp_ende = '1-1 2-4 3-2 4-4'
        alignment_gold = '1p1 1-5 2-3 3p4 4-2' # de-en always

        temp_align_path_ende = os.path.join(self.temp_dir.name, 'align_hyp_ende.txt')
        with open(temp_align_path_ende, 'w') as f:
            f.write(alignment_hyp_ende)
        
        temp_align_gold_path = os.path.join(self.temp_dir.name, 'align_gold.txt')
        with open(temp_align_gold_path, 'w') as f:
            f.write(alignment_gold)


        metrics_ende = calculate_word_saer(temp_align_path_ende, temp_align_gold_path, 'en-de')

        self.assertTrue(metrics_ende['aer'][0] == 1.0-((2.0+3.0)/(4.0+3.0)))

    def test_t_saer_s2t_ende_deen(self):
        alignment_hyp_ende = '1-1 2-4 3-2 4-4' # en-de
        total_len_ende = float(1 + 2 + 3 + 4)
        alignment_hyp_deen = '1-1 4-2 2-3 4-4 3-5' # de-en
        total_len_deen = float(1 + 4 + 2 + 4 + 3)
        alignment_gold = '1-1 1-5 2-3 3-4 4-2' # de-en always
        total_len_de = float(1 + 1 + 2 + 3 + 4)
        total_len_en = float(1 + 2 + 3 + 4 + 5)

        de_weights = {'000':{0:1, 1:2, 2:3, 3:4}} 
        en_weights = {'000':{0:1, 1:2, 2:3, 3:4, 4:5}} 
        
        temp_align_path_ende = os.path.join(self.temp_dir.name, 'align_hyp_ende.txt')
        with open(temp_align_path_ende, 'w') as f:
            f.write(alignment_hyp_ende)

        temp_align_path_deen = os.path.join(self.temp_dir.name, 'align_hyp_deen.txt')
        with open(temp_align_path_deen, 'w') as f:
            f.write(alignment_hyp_deen)
        
        temp_align_gold_path = os.path.join(self.temp_dir.name, 'align_gold.txt')
        with open(temp_align_gold_path, 'w') as f:
            f.write(alignment_gold)


        metrics_ende = calculate_time_saer(temp_align_path_ende, temp_align_gold_path, 'en-de', 's2t', en_weights, de_weights)
        metrics_deen = calculate_time_saer(temp_align_path_deen, temp_align_gold_path, 'de-en', 's2t', de_weights, en_weights)

        self.assertTrue(metrics_ende['aer'][0] == 1.0-((6.0 + 6.0)/(total_len_ende + total_len_en)))
        self.assertTrue(metrics_deen['aer'][0] == 1.0-((7.0 + 7.0)/(total_len_deen+total_len_de)))

    def test_t_saer_s2s_ende_deen(self):
        alignment_hyp_ende = '1-1 2-4 3-2 4-4' # en-de
        total_area_ende = float(1*1 + 2*4 + 3*2 + 4*4)
        alignment_hyp_deen = '1-1 4-2 2-3 4-4 3-5' # de-en
        total_area_deen = float(1*1 + 4*2 + 2*3 + 4*4 + 3*5)
        alignment_gold = '1-1 1-5 2-3 3-4 4-2' # de-en always
        total_area_gold = float(1*1 + 1*5 + 2*3 + 3*4 + 4*2)

        de_weights = {'000':{0:1, 1:2, 2:3, 3:4}} 
        en_weights = {'000':{0:1, 1:2, 2:3, 3:4, 4:5}} 
        
        temp_align_path_ende = os.path.join(self.temp_dir.name, 'align_hyp_ende.txt')
        with open(temp_align_path_ende, 'w') as f:
            f.write(alignment_hyp_ende)

        temp_align_path_deen = os.path.join(self.temp_dir.name, 'align_hyp_deen.txt')
        with open(temp_align_path_deen, 'w') as f:
            f.write(alignment_hyp_deen)
        
        temp_align_gold_path = os.path.join(self.temp_dir.name, 'align_gold.txt')
        with open(temp_align_gold_path, 'w') as f:
            f.write(alignment_gold)


        metrics_ende = calculate_time_saer(temp_align_path_ende, temp_align_gold_path, 'en-de', 's2s', en_weights, de_weights)
        metrics_deen = calculate_time_saer(temp_align_path_deen, temp_align_gold_path, 'de-en', 's2s', de_weights, en_weights)

        self.assertTrue(metrics_ende['aer'][0] == 1.0-((2*(1*1 + 4*2 + 2*3))/(total_area_ende + total_area_gold)))
        self.assertTrue(metrics_deen['aer'][0] == 1.0-((2*(1*1 + 2*3 + 4*2))/(total_area_deen + total_area_gold)))


if __name__ == '__main__':
    unittest.main()