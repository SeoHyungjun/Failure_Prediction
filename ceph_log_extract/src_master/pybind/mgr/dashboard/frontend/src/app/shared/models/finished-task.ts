import { Task } from './task';
import { TaskException } from './task-exception';

export class FinishedTask extends Task {
  begin_time: number;
  end_time: number;
  exception: TaskException;
  latency: number;
  progress: number;
  ret_value: any;
  success: boolean;

  errorMessage: string;
}
