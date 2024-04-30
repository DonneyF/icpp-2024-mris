from scheduler.scheduler import Scheduler
import numpy as np
import heapq
import tqdm

class BFEXECScheduler(Scheduler):
    # 10.1109/INFCOMW.2014.6849261
    def __repr__(self):
        return "BF-EXEC"

    def __phase0__(self, t, unscheduled_jobs, queue, R):
        num_processed_jobs = 0
        remaining_resources = np.zeros(shape=(len(self.machines), R))
        for i, machine in enumerate(self.machines):
            intervals = machine.intersecter.at(t)
            total_demand = np.add.reduce([interval.data.d for interval in intervals]) if intervals else np.zeros(R)
            remaining_resources[i, :] = machine.D - total_demand

        arrived_jobs = [job for job in unscheduled_jobs if job.r == t]

        for job in arrived_jobs:
            best_machines_indices = np.argsort(np.linalg.norm(remaining_resources, axis=1, ord=2))
            for i in best_machines_indices:
                machine = self.machines[i]
                if np.all(remaining_resources[i] - job.d >= 0):
                    # Set the first time we can feasible schedule
                    job.S = t
                    job.i = i
                    machine.add_job(job)
                    num_processed_jobs += 1

                    remaining_resources[i] -= job.d  # Update the demand since we've scheduled it to start at this time
                    break

        # Otherwise add the jobs to the queue
        for job in arrived_jobs:
            if job.S is None:
                heapq.heappush(queue, (job.p, job.id, job))

        return num_processed_jobs

    def __phase1__(self, t, phase1_machine_idx, queue, R):
        # Job left the system
        # Get the task with the shortest processing time in the queue and fit it to the recently freed server
        # Recurse until no more tasks fit
        machine = self.machines[phase1_machine_idx]
        num_processed_jobs = 0
        intervals = machine.intersecter.at(t)
        total_demand = np.add.reduce([interval.data.d for interval in intervals]) if intervals else np.zeros(R)
        heap_iterator = iter(heapq.nsmallest(len(queue), queue))
        for p, idx, job in heap_iterator:
            if (np.less_equal(total_demand + job.d, machine.D)).all():
                # Set the first time we can feasible schedule
                job.S = t
                job.i = phase1_machine_idx
                machine.add_job(job)
                num_processed_jobs += 1

                total_demand += job.d  # Update the demand since we've scheduled it to start at this time

        queue[:] = [tup for tup in queue if tup[2].S is None]

        return num_processed_jobs

    def process(self):
        try:
            R = self.jobs[0].d.size
        except AttributeError:
            R = 1

        num_processed_jobs = 0

        unscheduled_release_dates = [job.r for job in self.jobs]
        unscheduled_jobs = self.jobs.copy()

        t = min(unscheduled_release_dates)
        phase = 0
        queue = []
        phase1_machine_idx = 0 # Index of the machine whose job recently finished
        pbar = tqdm.tqdm(total=len(self.jobs), desc=self.__repr__(), position=self.id)
        while num_processed_jobs != len(self.jobs):
            num_phase_processed = 0
            if phase == 0:
                # New job arrived. Try to find an available machine
                num_phase_processed += self.__phase0__(t, unscheduled_jobs, queue, R)
            elif phase == 1:
                num_phase_processed += self.__phase1__(t, phase1_machine_idx, queue, R)
            elif phase == 2:
                num_phase_processed += self.__phase1__(t, phase1_machine_idx, queue, R)
                num_phase_processed += self.__phase0__(t, unscheduled_jobs, queue, R)

            num_processed_jobs += num_phase_processed
            pbar.update(num_phase_processed)

            # Need to advance to the next phase and keep track of the machine whose job was completed or job arrived
            earliest_completion_times = np.zeros(len(self.machines))
            for i, machine in enumerate(self.machines):
                intervals = machine.intersecter.at(t)
                completion_times = [j.end for j in intervals if j.end > t]
                earliest_completion_times[i] = min(completion_times) if completion_times else np.inf

            unscheduled_jobs[:] = [job for job in unscheduled_jobs if job.S is None]
            unscheduled_release_dates[:] = [ele for ele in unscheduled_release_dates if ele > t]

            earliest_completion_time = np.min(earliest_completion_times)
            earliest_release_date = min(unscheduled_release_dates) if unscheduled_release_dates else np.inf

            # Phase section with random tie breaking for machine capacity. On ties, phase 1 will reoccur
            if earliest_release_date > earliest_completion_time:
                phase = 1
                phase1_machine_idx = np.random.choice(np.flatnonzero(earliest_completion_times == earliest_completion_times.min()))
            elif earliest_release_date < earliest_completion_time:
                phase = 0
            else:
                # Perform phase 1, then phase 0 before updating the time
                phase = 2
                phase1_machine_idx = np.random.choice(np.flatnonzero(earliest_completion_times == earliest_completion_times.min()))

            t = min(earliest_release_date, earliest_completion_time)



