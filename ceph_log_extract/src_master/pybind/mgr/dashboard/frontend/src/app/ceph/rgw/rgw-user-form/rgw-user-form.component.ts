import { Component, OnDestroy, OnInit } from '@angular/core';
import {
  AbstractControl,
  AsyncValidatorFn,
  FormBuilder,
  FormGroup,
  ValidationErrors,
  Validators } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';

import * as _ from 'lodash';
import { BsModalService } from 'ngx-bootstrap';
import 'rxjs/add/observable/forkJoin';
import { Observable } from 'rxjs/Observable';

import { RgwUserService } from '../../../shared/api/rgw-user.service';
import { FormatterService } from '../../../shared/services/formatter.service';
import { CdValidators, isEmptyInputValue } from '../../../shared/validators/cd-validators';
import { RgwUserCapability } from '../models/rgw-user-capability';
import { RgwUserS3Key } from '../models/rgw-user-s3-key';
import { RgwUserSubuser } from '../models/rgw-user-subuser';
import { RgwUserSwiftKey } from '../models/rgw-user-swift-key';
import {
  RgwUserCapabilityModalComponent
} from '../rgw-user-capability-modal/rgw-user-capability-modal.component';
import {
  RgwUserS3KeyModalComponent
} from '../rgw-user-s3-key-modal/rgw-user-s3-key-modal.component';
import {
  RgwUserSubuserModalComponent
} from '../rgw-user-subuser-modal/rgw-user-subuser-modal.component';
import {
  RgwUserSwiftKeyModalComponent
} from '../rgw-user-swift-key-modal/rgw-user-swift-key-modal.component';

@Component({
  selector: 'cd-rgw-user-form',
  templateUrl: './rgw-user-form.component.html',
  styleUrls: ['./rgw-user-form.component.scss']
})
export class RgwUserFormComponent implements OnInit, OnDestroy {

  userForm: FormGroup;
  routeParamsSubscribe: any;
  editing = false;
  error = false;
  loading = false;
  submitObservables: Observable<Object>[] = [];

  subusers: RgwUserSubuser[] = [];
  s3Keys: RgwUserS3Key[] = [];
  swiftKeys: RgwUserSwiftKey[] = [];
  capabilities: RgwUserCapability[] = [];

  constructor(private formBuilder: FormBuilder,
              private route: ActivatedRoute,
              private router: Router,
              private rgwUserService: RgwUserService,
              private bsModalService: BsModalService) {
    this.createForm();
    this.listenToChanges();
  }

  createForm() {
    this.userForm = this.formBuilder.group({
      // General
      'user_id': [
        null,
        [Validators.required],
        [this.userIdValidator()]
      ],
      'display_name': [
        null,
        [Validators.required]
      ],
      'email': [
        null,
        [CdValidators.email]
      ],
      'max_buckets': [
        null,
        [Validators.min(0)]
      ],
      'suspended': [
        false
      ],
      // S3 key
      'generate_key': [
        true
      ],
      'access_key': [
        null,
        [CdValidators.requiredIf({'generate_key': false})]
      ],
      'secret_key': [
        null,
        [CdValidators.requiredIf({'generate_key': false})]
      ],
      // User quota
      'user_quota_enabled': [
        false
      ],
      'user_quota_max_size_unlimited': [
        true
      ],
      'user_quota_max_size': [
        null,
        [
          CdValidators.requiredIf({
            'user_quota_enabled': true,
            'user_quota_max_size_unlimited': false
          }),
          this.quotaMaxSizeValidator
        ]
      ],
      'user_quota_max_objects_unlimited': [
        true
      ],
      'user_quota_max_objects': [
        null,
        [
          Validators.min(0),
          CdValidators.requiredIf({
            'user_quota_enabled': true,
            'user_quota_max_objects_unlimited': false
          })
        ]
      ],
      // Bucket quota
      'bucket_quota_enabled': [
        false
      ],
      'bucket_quota_max_size_unlimited': [
        true
      ],
      'bucket_quota_max_size': [
        null,
        [
          CdValidators.requiredIf({
            'bucket_quota_enabled': true,
            'bucket_quota_max_size_unlimited': false
          }),
          this.quotaMaxSizeValidator
        ]
      ],
      'bucket_quota_max_objects_unlimited': [
        true
      ],
      'bucket_quota_max_objects': [
        null,
        [
          Validators.min(0),
          CdValidators.requiredIf({
            'bucket_quota_enabled': true,
            'bucket_quota_max_objects_unlimited': false
          })
        ]
      ]
    });
  }

  listenToChanges() {
    // Reset the validation status of various controls, especially those that are using
    // the 'requiredIf' validator. This is necessary because the controls itself are not
    // validated again if the status of their prerequisites have been changed.
    this.userForm.get('generate_key').valueChanges.subscribe(() => {
      ['access_key', 'secret_key'].forEach((path) => {
        this.userForm.get(path).updateValueAndValidity({onlySelf: true});
      });
    });
    this.userForm.get('user_quota_enabled').valueChanges.subscribe(() => {
      ['user_quota_max_size', 'user_quota_max_objects'].forEach((path) => {
        this.userForm.get(path).updateValueAndValidity({onlySelf: true});
      });
    });
    this.userForm.get('user_quota_max_size_unlimited').valueChanges.subscribe(() => {
      this.userForm.get('user_quota_max_size').updateValueAndValidity({onlySelf: true});
    });
    this.userForm.get('user_quota_max_objects_unlimited').valueChanges.subscribe(() => {
      this.userForm.get('user_quota_max_objects').updateValueAndValidity({onlySelf: true});
    });
    this.userForm.get('bucket_quota_enabled').valueChanges.subscribe(() => {
      ['bucket_quota_max_size', 'bucket_quota_max_objects'].forEach((path) => {
        this.userForm.get(path).updateValueAndValidity({onlySelf: true});
      });
    });
    this.userForm.get('bucket_quota_max_size_unlimited').valueChanges.subscribe(() => {
      this.userForm.get('bucket_quota_max_size').updateValueAndValidity({onlySelf: true});
    });
    this.userForm.get('bucket_quota_max_objects_unlimited').valueChanges.subscribe(() => {
      this.userForm.get('bucket_quota_max_objects').updateValueAndValidity({onlySelf: true});
    });
  }

  ngOnInit() {
    // Process route parameters.
    this.routeParamsSubscribe = this.route.params
      .subscribe((params: {uid: string}) => {
        if (!params.hasOwnProperty('uid')) {
          return;
        }
        this.loading = true;
        // Load the user data in 'edit' mode.
        this.editing = true;
        // Load the user and quota information.
        const observables = [];
        observables.push(this.rgwUserService.get(params.uid));
        observables.push(this.rgwUserService.getQuota(params.uid));
        Observable.forkJoin(observables)
          .subscribe((resp: any[]) => {
            this.loading = false;
            // Get the default values.
            const defaults = _.clone(this.userForm.value);
            // Extract the values displayed in the form.
            let value = _.pick(resp[0], _.keys(this.userForm.value));
            // Map the quota values.
            ['user', 'bucket'].forEach((type) => {
              const quota = resp[1][type + '_quota'];
              value[type + '_quota_enabled'] = quota.enabled;
              if (quota.max_size < 0) {
                value[type + '_quota_max_size_unlimited'] = true;
                value[type + '_quota_max_size'] = null;
              } else {
                value[type + '_quota_max_size_unlimited'] = false;
                value[type + '_quota_max_size'] = quota.max_size;
              }
              if (quota.max_objects < 0) {
                value[type + '_quota_max_size_unlimited'] = true;
                value[type + '_quota_max_size'] = null;
              } else {
                value[type + '_quota_max_objects_unlimited'] = false;
                value[type + '_quota_max_objects'] = quota.max_objects;
              }
            });
            // Merge with default values.
            value = _.merge(defaults, value);
            // Update the form.
            this.userForm.setValue(value);

            // Get the sub users.
            this.subusers = resp[0].subusers;

            // Get the keys.
            this.s3Keys = resp[0].keys;
            this.swiftKeys = resp[0].swift_keys;

            // Process the capabilities.
            const mapPerm = {'read, write': '*'};
            resp[0].caps.forEach((cap) => {
              if (cap.perm in mapPerm) {
                cap.perm = mapPerm[cap.perm];
              }
            });
            this.capabilities = resp[0].caps;
          }, (error) => {
            this.error = error;
          });
      });
  }

  ngOnDestroy() {
    this.routeParamsSubscribe.unsubscribe();
  }

  goToListView() {
    this.router.navigate(['/rgw/user']);
  }

  onSubmit() {
    // Exit immediately if the form isn't dirty.
    if (this.userForm.pristine) {
      this.goToListView();
    }
    if (this.editing) { // Edit
      if (this._isGeneralDirty()) {
        const args = this._getApiPostArgs();
        this.submitObservables.push(this.rgwUserService.post(args));
      }
    } else { // Add
      const args = this._getApiPutArgs();
      this.submitObservables.push(this.rgwUserService.put(args));
    }
    // Check if user quota has been modified.
    if (this._isUserQuotaDirty()) {
      const userQuotaArgs = this._getApiUserQuotaArgs();
      this.submitObservables.push(this.rgwUserService.putQuota(userQuotaArgs));
    }
    // Check if bucket quota has been modified.
    if (this._isBucketQuotaDirty()) {
      const bucketQuotaArgs = this._getApiBucketQuotaArgs();
      this.submitObservables.push(this.rgwUserService.putQuota(bucketQuotaArgs));
    }
    // Finally execute all observables.
    Observable.forkJoin(this.submitObservables)
      .subscribe(() => {
        this.goToListView();
      }, () => {
        // Reset the 'Submit' button.
        this.userForm.setErrors({'cdSubmitButton': true});
      });
  }

  /**
   * Validate the quota maximum size, e.g. 1096, 1K, 30M. Only integer numbers are valid,
   * something like 1.9M is not recognized as valid.
   */
  quotaMaxSizeValidator(control: AbstractControl): ValidationErrors | null {
    if (isEmptyInputValue(control.value)) {
      return null;
    }
    const m = RegExp('^(\\d+)\\s*(B|K(B|iB)?|M(B|iB)?|G(B|iB)?|T(B|iB)?)?$',
      'i').exec(control.value);
    if (m === null) {
      return {'quotaMaxSize': true};
    }
    const bytes = new FormatterService().toBytes(control.value);
    return (bytes < 1024) ? {'quotaMaxSize': true} : null;
  }

  /**
   * Validate the username.
   */
  userIdValidator(): AsyncValidatorFn {
    const rgwUserService = this.rgwUserService;
    return (control: AbstractControl): Promise<ValidationErrors | null> => {
      return new Promise((resolve) => {
        // Exit immediately if user has not interacted with the control yet
        // or the control value is empty.
        if (control.pristine || control.value === '') {
          resolve(null);
          return;
        }
        rgwUserService.exists(control.value)
          .subscribe((resp: boolean) => {
            if (!resp) {
              resolve(null);
            } else {
              resolve({'userIdExists': true});
            }
          });
      });
    };
  }

  /**
   * Add/Update a subuser.
   */
  setSubuser(subuser: RgwUserSubuser, index?: number) {
    if (_.isNumber(index)) { // Modify
      // Create an observable to modify the subuser when the form is submitted.
      this.submitObservables.push(this.rgwUserService.addSubuser(
        this.userForm.get('user_id').value, subuser.id, subuser.permissions,
        subuser.secret_key, subuser.generate_secret));
      this.subusers[index] = subuser;
    } else { // Add
      // Create an observable to add the subuser when the form is submitted.
      this.submitObservables.push(this.rgwUserService.addSubuser(
        this.userForm.get('user_id').value, subuser.id, subuser.permissions,
        subuser.secret_key, subuser.generate_secret));
      this.subusers.push(subuser);
      // Add a Swift key. If the secret key is auto-generated, then visualize
      // this to the user by displaying a notification instead of the key.
      this.swiftKeys.push({
        'user': subuser.id,
        'secret_key': subuser.generate_secret ?
          'Apply your changes first...' : subuser.secret_key
      });
    }
    // Mark the form as dirty to be able to submit it.
    this.userForm.markAsDirty();
  }

  /**
   * Delete a subuser.
   * @param {number} index The subuser to delete.
   */
  deleteSubuser(index: number) {
    const subuser = this.subusers[index];
    // Create an observable to delete the subuser when the form is submitted.
    this.submitObservables.push(this.rgwUserService.deleteSubuser(
      this.userForm.get('user_id').value, subuser.id));
    // Remove the associated S3 keys.
    this.s3Keys = this.s3Keys.filter((key) => {
      return key.user !== subuser.id;
    });
    // Remove the associated Swift keys.
    this.swiftKeys = this.swiftKeys.filter((key) => {
      return key.user !== subuser.id;
    });
    // Remove the subuser to update the UI.
    this.subusers.splice(index, 1);
    // Mark the form as dirty to be able to submit it.
    this.userForm.markAsDirty();
  }

  /**
   * Add/Update a capability.
   */
  setCapability(cap: RgwUserCapability, index?: number) {
    const uid = this.userForm.get('user_id').value;
    if (_.isNumber(index)) { // Modify
      const oldCap = this.capabilities[index];
      // Note, the RadosGW Admin OPS API does not support the modification of
      // user capabilities. Because of that it is necessary to delete it and
      // then to re-add the capability with its new value/permission.
      this.submitObservables.push(this.rgwUserService.deleteCapability(
        uid, oldCap.type, oldCap.perm));
      this.submitObservables.push(this.rgwUserService.addCapability(
        uid, cap.type, cap.perm));
      this.capabilities[index] = cap;
    } else { // Add
      // Create an observable to add the capability when the form is submitted.
      this.submitObservables.push(this.rgwUserService.addCapability(
        uid, cap.type, cap.perm));
      this.capabilities.push(cap);
    }
    // Mark the form as dirty to be able to submit it.
    this.userForm.markAsDirty();
  }

  /**
   * Delete the given capability:
   * - Delete it from the local array to update the UI
   * - Create an observable that will be executed on form submit
   * @param {number} index The capability to delete.
   */
  deleteCapability(index: number) {
    const cap = this.capabilities[index];
    // Create an observable to delete the capability when the form is submitted.
    this.submitObservables.push(this.rgwUserService.deleteCapability(
      this.userForm.get('user_id').value, cap.type, cap.perm));
    // Remove the capability to update the UI.
    this.capabilities.splice(index, 1);
    // Mark the form as dirty to be able to submit it.
    this.userForm.markAsDirty();
  }

  /**
   * Add/Update a S3 key.
   */
  setS3Key(key: RgwUserS3Key, index?: number) {
    if (_.isNumber(index)) { // Modify
      // Nothing to do here at the moment.
    } else { // Add
      // Create an observable to add the S3 key when the form is submitted.
      this.submitObservables.push(this.rgwUserService.addS3Key(
        this.userForm.get('user_id').value, key.user, key.access_key,
        key.secret_key, key.generate_key));
      // If the access and the secret key are auto-generated, then visualize
      // this to the user by displaying a notification instead of the key.
      this.s3Keys.push({
        'user': key.user,
        'access_key': key.generate_key ? 'Apply your changes first...' : key.access_key,
        'secret_key': key.generate_key ? 'Apply your changes first...' : key.secret_key
      });
    }
    // Mark the form as dirty to be able to submit it.
    this.userForm.markAsDirty();
  }

  /**
   * Delete a S3 key.
   * @param {number} index The S3 key to delete.
   */
  deleteS3Key(index: number) {
    const key = this.s3Keys[index];
    // Create an observable to delete the S3 key when the form is submitted.
    this.submitObservables.push(this.rgwUserService.deleteS3Key(
      this.userForm.get('user_id').value, key.access_key));
    // Remove the S3 key to update the UI.
    this.s3Keys.splice(index, 1);
    // Mark the form as dirty to be able to submit it.
    this.userForm.markAsDirty();
  }

  /**
   * Show the specified subuser in a modal dialog.
   * @param {number | undefined} index The subuser to show.
   */
  showSubuserModal(index?: number) {
    const uid = this.userForm.get('user_id').value;
    const modalRef = this.bsModalService.show(RgwUserSubuserModalComponent);
    if (_.isNumber(index)) { // Edit
      const subuser = this.subusers[index];
      modalRef.content.setEditing();
      modalRef.content.setValues(uid, subuser.id, subuser.permissions);
    } else { // Add
      modalRef.content.setEditing(false);
      modalRef.content.setValues(uid);
      modalRef.content.setSubusers(this.subusers);
    }
    modalRef.content.submitAction.subscribe((subuser: RgwUserSubuser) => {
      this.setSubuser(subuser, index);
    });
  }

  /**
   * Show the specified S3 key in a modal dialog.
   * @param {number | undefined} index The S3 key to show.
   */
  showS3KeyModal(index?: number) {
    const modalRef = this.bsModalService.show(RgwUserS3KeyModalComponent);
    if (_.isNumber(index)) { // View
      const key = this.s3Keys[index];
      modalRef.content.setViewing();
      modalRef.content.setValues(key.user, key.access_key, key.secret_key);
    } else { // Add
      const candidates = this._getS3KeyUserCandidates();
      modalRef.content.setViewing(false);
      modalRef.content.setUserCandidates(candidates);
      modalRef.content.submitAction.subscribe((key: RgwUserS3Key) => {
        this.setS3Key(key);
      });
    }
  }

  /**
   * Show the specified Swift key in a modal dialog.
   * @param {number} index The Swift key to show.
   */
  showSwiftKeyModal(index: number) {
    const modalRef = this.bsModalService.show(RgwUserSwiftKeyModalComponent);
    const key = this.swiftKeys[index];
    modalRef.content.setValues(key.user, key.secret_key);
  }

  /**
   * Show the specified capability in a modal dialog.
   * @param {number | undefined} index The S3 key to show.
   */
  showCapabilityModal(index?: number) {
    const modalRef = this.bsModalService.show(RgwUserCapabilityModalComponent);
    if (_.isNumber(index)) { // Edit
      const cap = this.capabilities[index];
      modalRef.content.setEditing();
      modalRef.content.setValues(cap.type, cap.perm);
    } else { // Add
      modalRef.content.setEditing(false);
      modalRef.content.setCapabilities(this.capabilities);
    }
    modalRef.content.submitAction.subscribe((cap: RgwUserCapability) => {
      this.setCapability(cap, index);
    });
  }

  /**
   * Check if the general user settings (display name, email, ...) have been modified.
   * @return {Boolean} Returns TRUE if the general user settings have been modified.
   */
  private _isGeneralDirty(): boolean {
    return [
      'display_name',
      'email',
      'max_buckets',
      'suspended'
    ].some((path) => {
      return this.userForm.get(path).dirty;
    });
  }

  /**
   * Check if the user quota has been modified.
   * @return {Boolean} Returns TRUE if the user quota has been modified.
   */
  private _isUserQuotaDirty(): boolean {
    return [
      'user_quota_enabled',
      'user_quota_max_size_unlimited',
      'user_quota_max_size',
      'user_quota_max_objects_unlimited',
      'user_quota_max_objects'
    ].some((path) => {
      return this.userForm.get(path).dirty;
    });
  }

  /**
   * Check if the bucket quota has been modified.
   * @return {Boolean} Returns TRUE if the bucket quota has been modified.
   */
  private _isBucketQuotaDirty(): boolean {
    return [
      'bucket_quota_enabled',
      'bucket_quota_max_size_unlimited',
      'bucket_quota_max_size',
      'bucket_quota_max_objects_unlimited',
      'bucket_quota_max_objects'
    ].some((path) => {
      return this.userForm.get(path).dirty;
    });
  }

  /**
   * Helper function to get the arguments of the API request when a new
   * user is created.
   */
  private _getApiPutArgs() {
    const result = {
      'uid': this.userForm.get('user_id').value,
      'display-name': this.userForm.get('display_name').value
    };
    const suspendedCtl = this.userForm.get('suspended');
    if (suspendedCtl.value) {
      _.extend(result, {'suspended': suspendedCtl.value});
    }
    const emailCtl = this.userForm.get('email');
    if (_.isString(emailCtl.value) && emailCtl.value.length > 0) {
      _.extend(result, { 'email': emailCtl.value });
    }
    const maxBucketsCtl = this.userForm.get('max_buckets');
    if (maxBucketsCtl.value > 0) {
      _.extend(result, {'max-buckets': maxBucketsCtl.value});
    }
    const generateKeyCtl = this.userForm.get('generate_key');
    if (!generateKeyCtl.value) {
      _.extend(result, {
        'access-key': this.userForm.get('access_key').value,
        'secret-key': this.userForm.get('secret_key').value
      });
    } else {
      _.extend(result, {'generate-key': true});
    }
    return result;
  }

  /**
   * Helper function to get the arguments for the API request when the user
   * configuration has been modified.
   */
  private _getApiPostArgs() {
    const result = {
      'uid': this.userForm.get('user_id').value
    };
    const argsMap = {
      'display-name': 'display_name',
      'email': 'email',
      'max-buckets': 'max_buckets',
      'suspended': 'suspended'
    };
    for (const key of Object.keys(argsMap)) {
      const ctl = this.userForm.get(argsMap[key]);
      if (ctl.dirty) {
        result[key] = ctl.value;
      }
    }
    return result;
  }

  /**
   * Helper function to get the arguments for the API request when the user
   * quota configuration has been modified.
   */
  private _getApiUserQuotaArgs(): object {
    const result = {
      'uid': this.userForm.get('user_id').value,
      'quota-type': 'user',
      'enabled': this.userForm.get('user_quota_enabled').value,
      'max-size-kb': -1,
      'max-objects': -1
    };
    if (!this.userForm.get('user_quota_max_size_unlimited').value) {
      // Convert the given value to bytes.
      const bytes = new FormatterService().toBytes(this.userForm.get(
        'user_quota_max_size').value);
      // Finally convert the value to KiB.
      result['max-size-kb'] = (bytes / 1024).toFixed(0) as any;
    }
    if (!this.userForm.get('user_quota_max_objects_unlimited').value) {
      result['max-objects'] = this.userForm.get('user_quota_max_objects').value;
    }
    return result;
  }

  /**
   * Helper function to get the arguments for the API request when the bucket
   * quota configuration has been modified.
   */
  private _getApiBucketQuotaArgs(): object {
    const result = {
      'uid': this.userForm.get('user_id').value,
      'quota-type': 'bucket',
      'enabled': this.userForm.get('bucket_quota_enabled').value,
      'max-size-kb': -1,
      'max-objects': -1
    };
    if (!this.userForm.get('bucket_quota_max_size_unlimited').value) {
      // Convert the given value to bytes.
      const bytes = new FormatterService().toBytes(this.userForm.get(
        'bucket_quota_max_size').value);
      // Finally convert the value to KiB.
      result['max-size-kb'] = (bytes / 1024).toFixed(0) as any;
    }
    if (!this.userForm.get('bucket_quota_max_objects_unlimited').value) {
      result['max-objects'] = this.userForm.get('bucket_quota_max_objects').value;
    }
    return result;
  }

  /**
   * Helper method to get the user candidates for S3 keys.
   * @returns {Array} Returns a list of user identifiers.
   */
  private _getS3KeyUserCandidates() {
    let result = [];
    // Add the current user id.
    const user_id = this.userForm.get('user_id').value;
    if (_.isString(user_id) && !_.isEmpty(user_id)) {
      result.push(user_id);
    }
    // Append the subusers.
    this.subusers.forEach((subUser) => {
      result.push(subUser.id);
    });
    // Note that it's possible to create multiple S3 key pairs for a user,
    // thus we append already configured users, too.
    this.s3Keys.forEach((key) => {
      result.push(key.user);
    });
    result = _.uniq(result);
    return result;
  }
}
