import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable()
export class RgwDaemonService {

  private url = 'api/rgw/daemon';

  constructor(private http: HttpClient) { }

  list() {
    return this.http.get(this.url);
  }

  get(id: string) {
    return this.http.get(`${this.url}/${id}`);
  }
}
