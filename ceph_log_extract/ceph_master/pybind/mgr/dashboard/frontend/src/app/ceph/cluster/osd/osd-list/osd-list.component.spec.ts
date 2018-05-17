import { HttpClientModule } from '@angular/common/http';
import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { TabsModule } from 'ngx-bootstrap/tabs';

import { ComponentsModule } from '../../../../shared/components/components.module';
import { DataTableModule } from '../../../../shared/datatable/datatable.module';
import { DimlessPipe } from '../../../../shared/pipes/dimless.pipe';
import { FormatterService } from '../../../../shared/services/formatter.service';
import { SharedModule } from '../../../../shared/shared.module';
import { PerformanceCounterModule } from '../../../performance-counter/performance-counter.module';
import { OsdDetailsComponent } from '../osd-details/osd-details.component';
import {
  OsdPerformanceHistogramComponent
} from '../osd-performance-histogram/osd-performance-histogram.component';
import { OsdListComponent } from './osd-list.component';

describe('OsdListComponent', () => {
  let component: OsdListComponent;
  let fixture: ComponentFixture<OsdListComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      imports: [
        HttpClientModule,
        PerformanceCounterModule,
        TabsModule.forRoot(),
        DataTableModule,
        ComponentsModule,
        SharedModule
      ],
      declarations: [
        OsdListComponent,
        OsdDetailsComponent,
        OsdPerformanceHistogramComponent
      ],
      providers: [ DimlessPipe, FormatterService]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(OsdListComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
