import numpy as np

from optint.model import linearSCM
from optint.acquisition import *
from optint.utils import compute_shd, learn_dag


def test_passive(problem, opts):
	A = []
	Prob = []	
	model = linearSCM(problem.DAG, {'pot_vec': 0, 'info_mat': 1})

	for _ in range(opts.T):
		a = np.random.uniform(-1,1,(problem.nnodes,1))
		a = a / np.linalg.norm(a)

		batch = problem.sample(a, opts.n)
		model.update_posterior(a, batch)

		A.append(a)
		Prob.append(model.prob_padded())

	return A, Prob


def test_active(problem, opts):
	A = []
	Prob = []
	model = linearSCM(problem.DAG, {'pot_vec': 0, 'info_mat': 1})
	
	# warm-up using randomly sampled interventions
	for _ in range(opts.W):
		a = np.random.uniform(-1,1,(problem.nnodes,1))
		a = a / np.linalg.norm(a)
		
		batch = problem.sample(a, opts.n)
		model.update_posterior(a, batch)

		A.append(a)
		Prob.append(model.prob_padded())

	acq = {
		'civ': civ_acq, 
		'greedy': greedy_acq,
		'maxv': maxv_acq,
		'cv': cv_acq,
		'mi': mi_acq,
		'ei': ei_acq,
		'ucb': ucb_acq
	}.get(opts.acq, None)
	assert acq is not None, "Unsupported functions!"

	for i in range(opts.T-opts.W):

		prob_pad = model.prob_padded()
		mean = np.array(prob_pad['mean'])
		var = np.array(prob_pad['var'])

		sigma_square = problem.sigma_square if opts.known_noise else np.zeros(problem.sigma_square.shape)

		if opts.acq == 'civ':
			acquisition = acq(sigma_square, mean, var, problem.mu_target, opts.n, opts.measure)
		else:	
			acquisition = acq(sigma_square, mean, var, problem.mu_target, opts.n)
		

		try:
			# try two initial points
			a_jitter = a.reshape(-1)*np.random.uniform(0.8,1.2,(problem.nnodes,))
			x01 = np.maximum(np.minimum(a_jitter, 1), -1)

			x02 = problem.mu_target - np.matmul(mean, problem.mu_target)
			x02 = x02.flatten()        
			x02 = x02 / np.linalg.norm(x02)

			a1 = acquisition.optimize(x0=x01)
			a2 = acquisition.optimize(x0=x02)
			if a1 is not None and a2 is not None:
				a1 = a1.reshape(-1,1)
				a2 = a2.reshape(-1,1)
				a = (a1, a2)[acquisition.evaluate(a1) > acquisition.evaluate(a2)]
			elif a1 is not None:
				a = a1.reshape(-1,1)
				print('2nd initialization (estimate) failed...')
			elif a2 is not None:
				a = a2.reshape(-1,1)
				print('1st initialization (last round) failed...')
			else:
				print('both initializations failed...')
		except UnboundLocalError:
			x0 = np.random.uniform(-1,1, problem.nnodes)
			x0 = x0 / np.linalg.norm(x0)
			try:
				a = acquisition.optimize(x0=x0).reshape(-1,1)
			except AttributeError:
				a = np.random.uniform(-1,1, problem.nnodes).reshape(-1,1)
				a = a / np.linalg.norm(a) 

		batch = problem.sample(a, opts.n)
		model.update_posterior(a, batch)

		if opts.acq == 'ei':
			acquisition.obj_min = min(acquisition.obj_min, np.linalg.norm(np.average(batch, axis=1)-problem.mu_target)**2)

		A.append(a)
		Prob.append(model.prob_padded())

	return A, Prob


def test_bayesian_active(problem, G_init, opts):
	"""
	Active learning with DAG re-learning.
	"""
	A = []
	Prob = []
	SHD = []

	current_dag = G_init
	model = linearSCM(current_dag, {'pot_vec': 0, 'info_mat': 1})

	all_batches = []
	all_interventions = []

	# Warm-up
	for _ in range(opts.W):
		a = np.random.uniform(-1,1,(problem.nnodes,1))
		a = a / np.linalg.norm(a)

		batch = problem.sample(a, opts.n)
		model.update_posterior(a, batch)

		all_batches.append(batch)
		all_interventions.append(a)

		A.append(a)
		Prob.append(model.prob_padded())
		SHD.append(compute_shd(current_dag, problem.DAG))

	for i in range(opts.T-opts.W):
		prob_pad = model.prob_padded()
		mean = np.array(prob_pad['mean'])
		var = np.array(prob_pad['var'])

		sigma_square = problem.sigma_square if opts.known_noise else np.zeros(problem.sigma_square.shape)

		acquisition = civ_acq(sigma_square, mean, var, problem.mu_target, opts.n, opts.measure)

		try:
			a_jitter = a.reshape(-1)*np.random.uniform(0.8,1.2,(problem.nnodes,))
			x01 = np.maximum(np.minimum(a_jitter, 1), -1)

			x02 = problem.mu_target - np.matmul(mean, problem.mu_target)
			x02 = x02.flatten()        
			x02 = x02 / np.linalg.norm(x02)

			a1 = acquisition.optimize(x0=x01)
			a2 = acquisition.optimize(x0=x02)

			if a1 is not None and a2 is not None:
				a1 = a1.reshape(-1,1)
				a2 = a2.reshape(-1,1)
				a = (a1, a2)[acquisition.evaluate(a1) > acquisition.evaluate(a2)]
			elif a1 is not None:
				a = a1.reshape(-1,1)
			elif a2 is not None:
				a = a2.reshape(-1,1)
			else:
				print("Both initializations failed...")
		except UnboundLocalError:
			x0 = np.random.uniform(-1,1, problem.nnodes)
			x0 = x0 / np.linalg.norm(x0)
			result = acquisition.optimize(x0=x0)
			if result is not None:
				a = result.reshape(-1,1)
			else:
				a = np.random.uniform(-1,1, problem.nnodes).reshape(-1,1)
				a = a / np.linalg.norm(a) 
		
		# Sample from true system
		batch = problem.sample(a, opts.n)

		# Store data
		all_batches.append(batch)
		all_interventions.append(a)

		# Re-learn DAG
		all_data = np.hstack(all_batches)

		current_dag = learn_dag(all_data, problem.nnodes)
		model = linearSCM(current_dag, {'pot_vec': 0, 'info_mat': 1})
		
		# Re-update with all data
		for batch_k, a_k in zip(all_batches, all_interventions):
			model.update_posterior(a_k, batch_k)

		A.append(a)
		Prob.append(model.prob_padded())
		SHD.append(compute_shd(current_dag, problem.DAG))

	return A, Prob, SHD