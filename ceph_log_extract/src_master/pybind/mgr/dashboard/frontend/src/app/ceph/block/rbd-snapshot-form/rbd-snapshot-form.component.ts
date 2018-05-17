import { Component, OnInit } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';

import { BsModalRef } from 'ngx-bootstrap';
import { Subject } from 'rxjs/Subject';

import { RbdService } from '../../../shared/api/rbd.service';
import { FinishedTask } from '../../../shared/models/finished-task';
import {
  NotificationService
} from '../../../shared/services/notification.service';
import { TaskManagerService } from '../../../shared/services/task-manager.service';

@Component({
  selector: 'cd-rbd-snapshot-form',
  templateUrl: './rbd-snapshot-form.component.html',
  styleUrls: ['./rbd-snapshot-form.component.scss']
})
export class RbdSnapshotFormComponent implements OnInit {

  poolName: string;
  imageName: string;
  snapName: string;

  snapshotForm: FormGroup;

  editing = false;

  public onSubmit: Subject<string>;

  constructor(public modalRef: BsModalRef,
              private rbdService: RbdService,
              private taskManagerService: TaskManagerService,
              private notificationService: NotificationService) {
    this.createForm();
  }

  createForm() {
    this.snapshotForm = new FormGroup({
      snapshotName: new FormControl('', {
        validators: [
          Validators.required
        ]
      })
    });
  }

  ngOnInit() {
    this.onSubmit = new Subject();
  }

  setSnapName(snapName) {
    this.snapName = snapName;
    this.snapshotForm.get('snapshotName').setValue(snapName);
    this.editing = true;
  }

  editAction() {
    const snapshotName = this.snapshotForm.get('snapshotName').value;
    const finishedTask = new FinishedTask();
    finishedTask.name = 'rbd/snap/edit';
    finishedTask.metadata = {
      'pool_name': this.poolName,
      'image_name': this.imageName,
      'snapshot_name': snapshotName
    };
    this.rbdService.renameSnapshot(this.poolName, this.imageName, this.snapName, snapshotName)
      .toPromise().then((resp) => {
        this.taskManagerService.subscribe(finishedTask.name, finishedTask.metadata,
          (asyncFinishedTask: FinishedTask) => {
            this.notificationService.notifyTask(asyncFinishedTask);
          });
        this.modalRef.hide();
        this.onSubmit.next(this.snapName);
      }).catch((resp) => {
        this.snapshotForm.setErrors({'cdSubmitButton': true});
      });
  }

  createAction() {
    const snapshotName = this.snapshotForm.get('snapshotName').value;
    const finishedTask = new FinishedTask();
    finishedTask.name = 'rbd/snap/create';
    finishedTask.metadata = {
      'pool_name': this.poolName,
      'image_name': this.imageName,
      'snapshot_name': snapshotName
    };
    this.rbdService.createSnapshot(this.poolName, this.imageName, snapshotName)
      .toPromise().then((resp) => {
        this.taskManagerService.subscribe(finishedTask.name, finishedTask.metadata,
          (asyncFinishedTask: FinishedTask) => {
            this.notificationService.notifyTask(asyncFinishedTask);
          });
        this.modalRef.hide();
        this.onSubmit.next(snapshotName);
      }).catch((resp) => {
        this.snapshotForm.setErrors({'cdSubmitButton': true});
      });
  }

  submit() {
    if (this.editing) {
      this.editAction();
    } else {
      this.createAction();
    }
  }
}
