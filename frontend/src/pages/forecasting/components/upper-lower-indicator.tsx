import { CanvasRenderingTarget2D } from 'fancy-canvas';
import {
	AutoscaleInfo,
	Coordinate,
	ISeriesPrimitive,
	ISeriesPrimitivePaneRenderer,
	ISeriesPrimitivePaneView,
	Logical,
	SeriesAttachedParameter,
	Time,
} from 'lightweight-charts';
import { PluginBase } from './plugin-base';
import { ClosestTimeIndexFinder } from '../helpers/closest-index';
import { UpperLowerInRange } from '../helpers/min-max-in-range';

interface BandRendererData {
	x: Coordinate | number;
	upper: Coordinate | number;
	lower: Coordinate | number;
}

class UpperLowerIndicatorPaneRenderer implements ISeriesPrimitivePaneRenderer {
	_viewData: BandViewData;
	constructor(data: BandViewData) {
		this._viewData = data;
	}
	draw() {}
	drawBackground(target: CanvasRenderingTarget2D) {
		const points: BandRendererData[] = this._viewData.data;
		target.useBitmapCoordinateSpace(scope => {
			const ctx = scope.context;
			ctx.scale(scope.horizontalPixelRatio, scope.verticalPixelRatio);

			ctx.strokeStyle = this._viewData.options.lineColor;
			ctx.lineWidth = this._viewData.options.lineWidth;
			ctx.beginPath();
			const region = new Path2D();
			const lines = new Path2D();
			region.moveTo(points[0].x, points[0].upper);
			lines.moveTo(points[0].x, points[0].upper);
			for (const point of points) {
				region.lineTo(point.x, point.upper);
				lines.lineTo(point.x, point.upper);
			}
			const end = points.length - 1;
			region.lineTo(points[end].x, points[end].lower);
			lines.moveTo(points[end].x, points[end].lower);
			for (let i = points.length - 2; i >= 0; i--) {
				region.lineTo(points[i].x, points[i].lower);
				lines.lineTo(points[i].x, points[i].lower);
			}
			region.lineTo(points[0].x, points[0].upper);
			region.closePath();
			ctx.stroke(lines);
			ctx.fillStyle = this._viewData.options.fillColor;
			ctx.fill(region);
		});
	}
}

interface BandViewData {
	data: BandRendererData[];
	options: Required<UpperLowerIndicatorOptions>;
}

class UpperLowerIndicatorPaneView implements ISeriesPrimitivePaneView {
	_source: UpperLowerIndicator;
	_data: BandViewData;

	constructor(source: UpperLowerIndicator) {
		this._source = source;
		this._data = {
			data: [],
			options: this._source._options,
		};
	}

	update() {
		if (!this._source.series || !this._source.chart) {
			// If series or chart is not available, don't update
			return;
		}
		const series = this._source.series;
		const timeScale = this._source.chart.timeScale();
		this._data.data = this._source._bandsData.map(d => {
			return {
				x: timeScale.timeToCoordinate(d.time) ?? -100,
				upper: series.priceToCoordinate(d.upper) ?? -100,
				lower: series.priceToCoordinate(d.lower) ?? -100,
			};
		});
	}

	renderer() {
		return new UpperLowerIndicatorPaneRenderer(this._data);
	}
}

export interface CustomBandData {
	time: Time;
	upper: number;
	lower: number;
}

export interface UpperLowerIndicatorOptions {
	lineColor?: string;
	fillColor?: string;
	lineWidth?: number;
}

const defaults: Required<UpperLowerIndicatorOptions> = {
	lineColor: 'rgb(25, 200, 100)',
	fillColor: 'rgba(25, 200, 100, 0.25)',
	lineWidth: 1,
};

export class UpperLowerIndicator extends PluginBase implements ISeriesPrimitive<Time> {
	_paneViews: UpperLowerIndicatorPaneView[];
	_bandsData: CustomBandData[] = [];
	_options: Required<UpperLowerIndicatorOptions>;
	_timeIndices: ClosestTimeIndexFinder<{ time: number }>;
	_upperLower: UpperLowerInRange<CustomBandData>;

	constructor(options: UpperLowerIndicatorOptions = {}) {
		super();
		this._options = { ...defaults, ...options };
		this._paneViews = [new UpperLowerIndicatorPaneView(this)];
		this._timeIndices = new ClosestTimeIndexFinder([]);
		this._upperLower = new UpperLowerInRange([]);
	}

	updateAllViews() {
		if (this.series && this.chart) {
			this._paneViews.forEach(pw => pw.update());
		}
	}

	paneViews() {
		return this._paneViews;
	}

	attached(p: SeriesAttachedParameter<Time>): void {
		super.attached(p);
		if (this.series && this.chart) {
			this.updateAllViews();
		}
	}

	setData(data: CustomBandData[]) {
		this._bandsData = data;
		this._timeIndices = new ClosestTimeIndexFinder(this._bandsData as { time: number }[]);
		this._upperLower = new UpperLowerInRange(this._bandsData, 4);
		if (this.series && this.chart) {
			this.updateAllViews();
		}
	}

	autoscaleInfo(startTimePoint: Logical, endTimePoint: Logical): AutoscaleInfo {
		const ts = this.chart.timeScale();
		const startTime = (ts.coordinateToTime(
			ts.logicalToCoordinate(startTimePoint) ?? 0
		) ?? 0) as number;
		const endTime = (ts.coordinateToTime(
			ts.logicalToCoordinate(endTimePoint) ?? 5000000000
		) ?? 5000000000) as number;
		const startIndex = this._timeIndices.findClosestIndex(startTime, 'left');
		const endIndex = this._timeIndices.findClosestIndex(endTime, 'right');
		const range = this._upperLower.getMinMax(startIndex, endIndex);
		return {
			priceRange: {
				minValue: range.lower,
				maxValue: range.upper,
			},
		};
	}
}