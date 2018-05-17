import { Component, EventEmitter, Output } from '@angular/core';
import {
  AbstractControl,
  FormBuilder,
  FormGroup,
  ValidationErrors,
  ValidatorFn,
  Validators
} from '@angular/forms';

import * as _ from 'lodash';
import { BsModalRef } from 'ngx-bootstrap/modal/bs-modal-ref.service';

import { CdValidators, isEmptyInputValue } from '../../../shared/validators/cd-validators';
import { RgwUserSubuser } from '../models/rgw-user-subuser';

@Component({
  selector: 'cd-rgw-user-subuser-modal',
  templateUrl: './rgw-user-subuser-modal.component.html',
  styleUrls: ['./rgw-user-subuser-modal.component.scss']
})
export class RgwUserSubuserModalComponent {

  /**
   * The event that is triggered when the 'Add' or 'Update' button
   * has been pressed.
   */
  @Output() submitAction = new EventEmitter();

  formGroup: FormGroup;
  editing = true;
  subusers: RgwUserSubuser[] = [];

  constructor(private formBuilder: FormBuilder,
              public bsModalRef: BsModalRef) {
    this.createForm();
    this.listenToChanges();
  }

  createForm() {
    this.formGroup = this.formBuilder.group({
      'uid': [
        null
      ],
      'subuid': [
        null,
        [
          Validators.required,
          this.subuserValidator()
        ]
      ],
      'perm': [
        null,
        [Validators.required]
      ],
      // Swift key
      'generate_secret': [
        true
      ],
      'secret_key': [
        null,
        [CdValidators.requiredIf({'generate_secret': false})]
      ]
    });
  }

  listenToChanges() {
    // Reset the validation status of various controls, especially those that are using
    // the 'requiredIf' validator. This is necessary because the controls itself are not
    // validated again if the status of their prerequisites have been changed.
    this.formGroup.get('generate_secret').valueChanges.subscribe(() => {
      ['secret_key'].forEach((path) => {
        this.formGroup.get(path).updateValueAndValidity({onlySelf: true});
      });
    });
  }

  /**
   * Validates whether the subuser already exists.
   */
  subuserValidator(): ValidatorFn {
    const self = this;
    return (control: AbstractControl): ValidationErrors | null => {
      if (self.editing) {
        return null;
      }
      if (isEmptyInputValue(control.value)) {
        return null;
      }
      const found = self.subusers.some((subuser) => {
        return _.isEqual(self.getSubuserName(subuser.id), control.value);
      });
      return found ? {'subuserIdExists': true} : null;
    };
  }

  /**
   * Get the subuser name.
   * Examples:
   *   'johndoe' => 'johndoe'
   *   'janedoe:xyz' => 'xyz'
   * @param {string} value The value to process.
   * @returns {string} Returns the user ID.
   */
  private getSubuserName(value: string) {
    if (_.isEmpty(value)) {
      return value;
    }
    const matches = value.match(/([^:]+)(:(.+))?/);
    return _.isUndefined(matches[3]) ? matches[1] : matches[3];
  }

  /**
   * Set the 'editing' flag. If set to TRUE, the modal dialog is in 'Edit' mode,
   * otherwise in 'Add' mode. According to the mode the dialog and its controls
   * behave different.
   * @param {boolean} viewing
   */
  setEditing(editing: boolean = true) {
    this.editing = editing;
  }

  /**
   * Set the values displayed in the dialog.
   */
  setValues(uid: string, subuser: string = '', permissions: string = '') {
    this.formGroup.setValue({
      'uid': uid,
      'subuid': this.getSubuserName(subuser),
      'perm': permissions,
      'generate_secret': true,
      'secret_key': null
    });
  }

  /**
   * Set the current capabilities of the user.
   */
  setSubusers(subusers: RgwUserSubuser[]) {
    this.subusers = subusers;
  }

  onSubmit() {
    // Get the values from the form and create an object that is sent
    // by the triggered submit action event.
    const values = this.formGroup.value;
    const subuser = new RgwUserSubuser;
    subuser.id = `${values.uid}:${values.subuid}`;
    subuser.permissions = values.perm;
    subuser.generate_secret = values.generate_secret;
    subuser.secret_key = values.secret_key;
    this.submitAction.emit(subuser);
    this.bsModalRef.hide();
  }
}
