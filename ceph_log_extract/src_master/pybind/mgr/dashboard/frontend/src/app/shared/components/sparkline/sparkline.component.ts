import { Component, ElementRef, OnChanges, OnInit, SimpleChanges, ViewChild } from '@angular/core';
import { Input } from '@angular/core';

import { ChartTooltip } from '../../../shared/models/chart-tooltip';

@Component({
  selector: 'cd-sparkline',
  templateUrl: './sparkline.component.html',
  styleUrls: ['./sparkline.component.scss']
})
export class SparklineComponent implements OnInit, OnChanges {
  @ViewChild('sparkCanvas') chartCanvasRef: ElementRef;
  @ViewChild('sparkTooltip') chartTooltipRef: ElementRef;

  @Input() data: any;
  @Input()
  style = {
    height: '30px',
    width: '100px'
  };

  public colors: Array<any> = [
    {
      backgroundColor: 'rgba(40,140,234,0.2)',
      borderColor: 'rgba(40,140,234,1)',
      pointBackgroundColor: 'rgba(40,140,234,1)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'rgba(40,140,234,0.8)'
    }
  ];

  options = {
    animation: {
      duration: 0
    },
    responsive: true,
    maintainAspectRatio: false,
    legend: {
      display: false
    },
    elements: {
      line: {
        borderWidth: 1
      }
    },
    tooltips: {
      enabled: false,
      mode: 'index',
      intersect: false,
      custom: undefined
    },
    scales: {
      yAxes: [
        {
          display: false
        }
      ],
      xAxes: [
        {
          display: false
        }
      ]
    }
  };

  public datasets: Array<any> = [
    {
      data: []
    }
  ];

  public labels: Array<any> = [];

  constructor() {}

  ngOnInit() {
    const getStyleTop = (tooltip, positionY) => {
      return (tooltip.caretY - tooltip.height - tooltip.yPadding - 5) + 'px';
    };

    const getStyleLeft = (tooltip, positionX) => {
      return positionX + tooltip.caretX + 'px';
    };

    const chartTooltip = new ChartTooltip(
      this.chartCanvasRef,
      this.chartTooltipRef,
      getStyleLeft,
      getStyleTop
    );

    chartTooltip.customColors = {
      backgroundColor: this.colors[0].pointBackgroundColor,
      borderColor: this.colors[0].pointBorderColor
    };

    this.options.tooltips.custom = tooltip => {
      chartTooltip.customTooltips(tooltip);
    };
  }

  ngOnChanges(changes: SimpleChanges) {
    this.datasets[0].data = changes['data'].currentValue;
    this.datasets = [...this.datasets];
    this.labels = [...Array(changes['data'].currentValue.length)];
  }
}
