from utils import get_sequences


def get_test_score():
	preddf = pd.read_csv('data/cagi4_mpra/4-eQTL-causal_SNPs_dataset2.txt', delimiter='\t')

	# assert 'ref_sequence' in df.columns, 'DataFrame must contain sequences already'